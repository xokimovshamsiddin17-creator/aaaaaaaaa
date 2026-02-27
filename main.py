# main.py - To'liq Telegram bot (aiogram 3.x) with multi-user support

import asyncio
import logging
import sqlite3
import random
import string
import re
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, BaseMiddleware
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.exceptions import TelegramBadRequest

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
SUPER_ADMIN_IDS = [int(id_) for id_ in os.getenv("ADMIN_IDS", "").split(",") if id_]
DATABASE_PATH = os.getenv("DATABASE_PATH", "bot_database.db")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== DATABASE CLASS ====================

class Database:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER UNIQUE NOT NULL,
                    username TEXT,
                    first_name TEXT NOT NULL,
                    last_name TEXT,
                    is_super_admin INTEGER DEFAULT 0,
                    is_admin INTEGER DEFAULT 1,
                    max_files INTEGER DEFAULT 10,
                    max_channels INTEGER DEFAULT 3,
                    max_file_size INTEGER DEFAULT 52428800,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    subscription_expires TIMESTAMP
                )
            ''')
            
            # Files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    views INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (telegram_id)
                )
            ''')
            
            # File items table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    telegram_file_id TEXT NOT NULL,
                    file_name TEXT,
                    file_size INTEGER,
                    FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
                )
            ''')
            
            # Channels table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER UNIQUE NOT NULL,
                    channel_username TEXT,
                    channel_title TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (telegram_id)
                )
            ''')
            
            # User statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL,
                    files_uploaded INTEGER DEFAULT 0,
                    files_downloaded INTEGER DEFAULT 0,
                    total_views INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (telegram_id)
                )
            ''')
            
            # Downloads log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS downloads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files (id),
                    FOREIGN KEY (user_id) REFERENCES users (telegram_id)
                )
            ''')
            
            # Subscriptions log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (telegram_id),
                    FOREIGN KEY (channel_id) REFERENCES channels (id)
                )
            ''')
            
            # Super adminlarni qo'shish
            for admin_id in SUPER_ADMIN_IDS:
                cursor.execute('''
                    INSERT OR IGNORE INTO users 
                    (telegram_id, is_super_admin, is_admin, max_files, max_channels, max_file_size, first_name)
                    VALUES (?, 1, 1, 999999, 999999, 1073741824, 'Super Admin')
                ''', (admin_id,))
                
                cursor.execute('''
                    INSERT OR IGNORE INTO user_stats (user_id)
                    VALUES (?)
                ''', (admin_id,))
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    # ========== USER METHODS ==========
    
    def add_user(self, telegram_id: int, first_name: str, username: Optional[str] = None, 
                 last_name: Optional[str] = None) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE telegram_id = ?', (telegram_id,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute('''
                    UPDATE users 
                    SET last_active = CURRENT_TIMESTAMP,
                        username = COALESCE(?, username),
                        first_name = COALESCE(?, first_name),
                        last_name = COALESCE(?, last_name)
                    WHERE telegram_id = ?
                ''', (username, first_name, last_name, telegram_id))
            else:
                cursor.execute('''
                    INSERT INTO users (
                        telegram_id, username, first_name, last_name, 
                        is_admin, max_files, max_channels, max_file_size
                    )
                    VALUES (?, ?, ?, ?, 1, 10, 3, 52428800)
                ''', (telegram_id, username, first_name, last_name))
                
                cursor.execute('''
                    INSERT OR IGNORE INTO user_stats (user_id)
                    VALUES (?)
                ''', (telegram_id,))
            
            conn.commit()
    
    def get_user(self, telegram_id: int) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE telegram_id = ?', (telegram_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def is_super_admin(self, telegram_id: int) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT is_super_admin FROM users WHERE telegram_id = ?', (telegram_id,))
            row = cursor.fetchone()
            return bool(row['is_super_admin']) if row else False
    
    def get_user_limits(self, telegram_id: int) -> Dict[str, int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT max_files, max_channels, max_file_size 
                FROM users WHERE telegram_id = ?
            ''', (telegram_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {'max_files': 10, 'max_channels': 3, 'max_file_size': 52428800}
    
    def get_user_files_count(self, telegram_id: int) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) as count FROM files 
                WHERE user_id = ? AND is_active = 1
            ''', (telegram_id,))
            row = cursor.fetchone()
            return row['count']
    
    def get_user_channels_count(self, telegram_id: int) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) as count FROM channels 
                WHERE user_id = ? AND is_active = 1
            ''', (telegram_id,))
            row = cursor.fetchone()
            return row['count']
    
    # ========== FILE METHODS ==========
    
    def create_file(self, code: str, user_id: int, description: Optional[str] = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files (code, user_id, description)
                VALUES (?, ?, ?)
            ''', (code, user_id, description))
            conn.commit()
            
            cursor.execute('''
                UPDATE user_stats 
                SET files_uploaded = files_uploaded + 1,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
            
            return cursor.lastrowid
    
    def add_file_item(self, file_id: int, file_type: str, telegram_file_id: str, 
                      file_name: Optional[str] = None, file_size: Optional[int] = None) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO file_items (file_id, file_type, telegram_file_id, file_name, file_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (file_id, file_type, telegram_file_id, file_name, file_size))
            conn.commit()
    
    def get_file_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.*, u.telegram_id as owner_id, u.username as owner_username,
                       u.first_name as owner_first_name
                FROM files f
                JOIN users u ON f.user_id = u.telegram_id
                WHERE f.code = ? AND f.is_active = 1
            ''', (code.upper(),))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_file_items(self, file_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM file_items WHERE file_id = ?', (file_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_user_files(self, user_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.*, COUNT(fi.id) as items_count 
                FROM files f
                LEFT JOIN file_items fi ON f.id = fi.file_id
                WHERE f.user_id = ? AND f.is_active = 1
                GROUP BY f.id
                ORDER BY f.created_at DESC
            ''', (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def increment_file_views(self, file_id: int) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE files SET views = views + 1 WHERE id = ?
            ''', (file_id,))
            
            cursor.execute('SELECT user_id FROM files WHERE id = ?', (file_id,))
            file = cursor.fetchone()
            if file:
                cursor.execute('''
                    UPDATE user_stats 
                    SET total_views = total_views + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (file['user_id'],))
            
            conn.commit()
    
    def log_download(self, file_id: int, user_id: int) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO downloads (file_id, user_id)
                VALUES (?, ?)
            ''', (file_id, user_id))
            
            cursor.execute('SELECT user_id FROM files WHERE id = ?', (file_id,))
            file = cursor.fetchone()
            if file:
                cursor.execute('''
                    UPDATE user_stats 
                    SET files_downloaded = files_downloaded + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (file['user_id'],))
            
            conn.commit()
    
    def delete_file(self, file_id: int, user_id: int) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE files SET is_active = 0 
                WHERE id = ? AND user_id = ?
            ''', (file_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_owner_channels(self, owner_id: int) -> List[Dict[str, Any]]:
        """Fayl egasining faol kanallarini olish"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM channels 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY added_at DESC
            ''', (owner_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # ========== CHANNEL METHODS ==========
    
    def add_channel(self, channel_id: int, channel_title: str, user_id: int, 
                    channel_username: Optional[str] = None) -> bool:
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO channels 
                    (channel_id, channel_username, channel_title, user_id, is_active)
                    VALUES (?, ?, ?, ?, 1)
                ''', (channel_id, channel_username, channel_title, user_id))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error adding channel: {e}")
                return False
    
    def remove_channel(self, channel_id: int, user_id: int) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE channels SET is_active = 0 
                WHERE channel_id = ? AND user_id = ?
            ''', (channel_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_user_channels(self, user_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM channels 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY added_at DESC
            ''', (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def log_subscription_check(self, user_id: int, channel_id: int) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO subscriptions (user_id, channel_id)
                VALUES (?, ?)
            ''', (user_id, channel_id))
            conn.commit()
    
    def has_recent_subscription_check(self, user_id: int, channel_id: int) -> bool:
        """Oxirgi 5 daqiqada tekshirilganligini aniqlash"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            five_minutes_ago = (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                SELECT COUNT(*) as count FROM subscriptions
                WHERE user_id = ? AND channel_id = ? AND checked_at > ?
            ''', (user_id, channel_id, five_minutes_ago))
            row = cursor.fetchone()
            return row['count'] > 0
    
    # ========== STATS METHODS ==========
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_stats WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                stats = dict(row)
            else:
                stats = {
                    'files_uploaded': 0,
                    'files_downloaded': 0,
                    'total_views': 0
                }
            
            stats['current_files'] = self.get_user_files_count(user_id)
            stats['current_channels'] = self.get_user_channels_count(user_id)
            
            return stats

# ==================== STATES ====================

class UserStates(StatesGroup):
    waiting_for_files = State()
    waiting_for_code = State()
    waiting_for_channel = State()

# ==================== UTILITIES ====================

class CodeGenerator:
    @staticmethod
    def generate_code(length: int = 8) -> str:
        characters = string.ascii_uppercase + string.digits
        return ''.join(random.choices(characters, k=length))
    
    @staticmethod
    def validate_code(code: str) -> bool:
        return bool(re.match(r'^[A-Z0-9]{8}$', code.upper()))

class ChannelChecker:
    def __init__(self, bot: Bot):
        self.bot = bot
    
    async def check_subscription(self, user_id: int, channels: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        not_subscribed = []
        subscribed = []
        
        for channel in channels:
            try:
                chat_member = await self.bot.get_chat_member(
                    chat_id=channel['channel_id'], 
                    user_id=user_id
                )
                
                if chat_member.status in ['left', 'kicked']:
                    not_subscribed.append(channel)
                else:
                    subscribed.append(channel)
                    
            except Exception as e:
                logger.error(f"Error checking subscription for channel {channel['channel_id']}: {e}")
                not_subscribed.append(channel)
        
        return not_subscribed, subscribed
    
    async def get_channel_info(self, channel_identifier: str) -> Optional[Dict[str, Any]]:
        try:
            if channel_identifier.startswith('@'):
                channel_identifier = channel_identifier[1:]
            
            if 't.me/' in channel_identifier:
                channel_identifier = channel_identifier.split('t.me/')[-1].split('/')[0]
            
            chat = await self.bot.get_chat(f"@{channel_identifier}")
            
            try:
                bot_member = await self.bot.get_chat_member(chat.id, self.bot.id)
                if bot_member.status not in ['administrator', 'creator']:
                    logger.warning(f"Bot is not admin in channel {channel_identifier}")
                    return None
            except:
                logger.warning(f"Could not check bot status in {channel_identifier}")
                return None
            
            return {
                'id': chat.id,
                'username': chat.username,
                'title': chat.title,
                'type': chat.type
            }
            
        except TelegramBadRequest as e:
            logger.error(f"Telegram error getting channel info: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting channel info: {e}")
            return None

def extract_channel_username(text: str) -> Optional[str]:
    text = text.strip()
    
    if text.startswith('@'):
        return text[1:]
    
    telegram_pattern = r'(?:https?://)?(?:t\.me|telegram\.me)/([a-zA-Z0-9_]+)'
    match = re.search(telegram_pattern, text)
    if match:
        return match.group(1)
    
    if re.match(r'^[a-zA-Z0-9_]{5,}$', text):
        return text
    
    return None

def format_size(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "Noma'lum"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def format_number(num: int) -> str:
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    else:
        return f"{num/1000000:.1f}M"

# ==================== KEYBOARDS ====================

def get_main_menu_keyboard(is_super_admin: bool = False) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="📤 Fayl yuklash", callback_data="upload_file"),
         InlineKeyboardButton(text="📂 Mening fayllarim", callback_data="my_files")],
        [InlineKeyboardButton(text="➕ Kanal qo'shish", callback_data="add_channel"),
         InlineKeyboardButton(text="➖ Kanal o'chirish", callback_data="remove_channel")],
        [InlineKeyboardButton(text="📋 Kanallarim", callback_data="list_my_channels"),
         InlineKeyboardButton(text="📊 Mening statistikam", callback_data="my_stats")],
        [InlineKeyboardButton(text="🔑 Kod yuborish", callback_data="send_code")],
        [InlineKeyboardButton(text="⚙️ Sozlamalar", callback_data="settings")]
    ]
    
    if is_super_admin:
        buttons.append([InlineKeyboardButton(text="👑 Super Admin Panel", callback_data="super_admin")])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_back_keyboard() -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")]]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_cancel_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="❌ Bekor qilish", callback_data="cancel")],
        [InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_subscription_keyboard(channels: List[Dict[str, Any]], code: str) -> InlineKeyboardMarkup:
    buttons = []
    for channel in channels:
        if channel['channel_username']:
            url = f"https://t.me/{channel['channel_username']}"
        else:
            url = f"https://t.me/c/{str(channel['channel_id'])[4:]}"
        
        display_name = channel['channel_title']
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        
        buttons.append([InlineKeyboardButton(text=f"📢 {display_name}", url=url)])
    
    buttons.append([InlineKeyboardButton(text="✅ Obunani tekshirish", callback_data=f"check_sub_{code}")])
    buttons.append([InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_files_keyboard(files: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    buttons = []
    for file in files:
        buttons.append([InlineKeyboardButton(
            text=f"📁 {file['code']} ({file['items_count']} fayl | 👁 {format_number(file['views'])})",
            callback_data=f"view_file_{file['id']}"
        )])
    buttons.append([InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_file_actions_keyboard(file_id: int, is_owner: bool = False) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="📥 Yuklab olish", callback_data=f"download_file_{file_id}")]
    ]
    
    if is_owner:
        buttons.append([InlineKeyboardButton(text="🗑 O'chirish", callback_data=f"delete_file_{file_id}")])
    
    buttons.append([InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_channels_keyboard(channels: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    buttons = []
    for channel in channels:
        display_name = channel['channel_title']
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        
        buttons.append([InlineKeyboardButton(
            text=f"📢 {display_name}",
            callback_data=f"view_channel_{channel['channel_id']}"
        )])
    
    buttons.append([InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_channel_actions_keyboard(channel_id: int) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="➖ O'chirish", callback_data=f"remove_my_channel_{channel_id}")],
        [InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_settings_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="📊 Limitlarni ko'rish", callback_data="view_limits")],
        [InlineKeyboardButton(text="ℹ️ Bot haqida", callback_data="about_bot")],
        [InlineKeyboardButton(text="👨‍💻 Yaratuvchi", callback_data="about_creator")],
        [InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_super_admin_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="👥 Foydalanuvchilar", callback_data="sa_users")],
        [InlineKeyboardButton(text="📊 Umumiy statistika", callback_data="sa_stats")],
        [InlineKeyboardButton(text="📢 Barcha kanallar", callback_data="sa_all_channels")],
        [InlineKeyboardButton(text="🏠 Bosh menyu", callback_data="back_to_main")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

# ==================== MIDDLEWARES ====================

class SubscriptionMiddleware(BaseMiddleware):
    def __init__(self, db: Database, channel_checker: ChannelChecker):
        self.db = db
        self.channel_checker = channel_checker
        super().__init__()
    
    async def __call__(self, handler, event: Message | CallbackQuery, data: Dict[str, Any]):
        if self.db.is_super_admin(event.from_user.id):
            return await handler(event, data)
        return await handler(event, data)

# ==================== HANDLERS ====================

class BotHandlers:
    def __init__(self, bot: Bot, dp: Dispatcher, db: Database, channel_checker: ChannelChecker):
        self.bot = bot
        self.dp = dp
        self.db = db
        self.channel_checker = channel_checker
        self.setup_handlers()
    
    async def handle_code(self, message: Message, code: str, state: FSMContext):
        """Kodni qayta ishlash - TO'G'RILANGAN VERSIYA"""
        user_id = message.from_user.id
        file_data = self.db.get_file_by_code(code)
        
        if not file_data:
            await message.answer(
                "❌ *Bunday kod mavjud emas!*",
                parse_mode="Markdown"
            )
            await state.clear()
            return
        
        # Fayl egasining kanallarini olish
        owner_channels = self.db.get_owner_channels(file_data['user_id'])
        
        # Agar fayl egasining kanallari bo'lsa
        if owner_channels:
            not_subscribed, subscribed = await self.channel_checker.check_subscription(user_id, owner_channels)
            
            if not_subscribed:
                await message.answer(
                    "❗️ *Faylni olish uchun quyidagi kanallarga obuna bo'ling:*\n\n"
                    "Obuna bo'lgach '✅ Obunani tekshirish' tugmasini bosing.",
                    parse_mode="Markdown",
                    reply_markup=get_subscription_keyboard(not_subscribed, code)
                )
                await state.update_data(pending_code=code, owner_id=file_data['user_id'])
                return
        
        # Agar obuna bo'lgan bo'lsa yoki kanallar bo'lmasa
        await self.send_file_content(message, file_data, user_id)
        await state.clear()
    
    async def send_file_content(self, message: Message, file_data: Dict[str, Any], user_id: int):
        """Fayllarni yuborish"""
        file_items = self.db.get_file_items(file_data['id'])
        
        if not file_items:
            await message.answer(
                "❌ *Bu kodda fayllar mavjud emas!*",
                parse_mode="Markdown"
            )
            return
        
        # Increment views and log download
        self.db.increment_file_views(file_data['id'])
        self.db.log_download(file_data['id'], user_id)
        
        # Fayl egasiga xabar
        if file_data['user_id'] != user_id:
            try:
                user_info = await self.bot.get_chat(user_id)
                user_name = user_info.username or user_info.first_name
                
                await self.bot.send_message(
                    file_data['user_id'],
                    f"📥 *Sizning faylingiz yuklab olindi!*\n\n"
                    f"Kod: `{file_data['code']}`\n"
                    f"Foydalanuvchi: @{user_name if user_info.username else user_name}\n"
                    f"Ko'rishlar: {file_data['views'] + 1}",
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Error notifying owner: {e}")
        
        await message.answer(
            f"📥 *{len(file_items)} ta fayl yuborilmoqda...*",
            parse_mode="Markdown"
        )
        
        for item in file_items:
            try:
                caption = f"📁 {item['file_name'] or 'Fayl'}"
                if item['file_size']:
                    caption += f" | {format_size(item['file_size'])}"
                
                if item['file_type'] == 'photo':
                    await message.answer_photo(
                        item['telegram_file_id'],
                        caption=caption
                    )
                elif item['file_type'] == 'video':
                    await message.answer_video(
                        item['telegram_file_id'],
                        caption=caption
                    )
                else:
                    await message.answer_document(
                        item['telegram_file_id'],
                        caption=caption
                    )
            except Exception as e:
                logger.error(f"Error sending file: {e}")
                try:
                    await message.answer_document(
                        item['telegram_file_id'],
                        caption="📁 Fayl"
                    )
                except:
                    pass
        
        await message.answer(
            "✅ *Barcha fayllar yuborildi!*",
            parse_mode="Markdown"
        )
    
    def setup_handlers(self):
        
        @self.dp.message(CommandStart())
        async def cmd_start(message: Message):
            user = message.from_user
            self.db.add_user(
                telegram_id=user.id,
                first_name=user.first_name,
                username=user.username,
                last_name=user.last_name
            )
            
            limits = self.db.get_user_limits(user.id)
            stats = self.db.get_user_stats(user.id)
            
            welcome_text = (
                f"✨ *Assalomu alaykum, {user.first_name}!* ✨\n\n"
                f"📊 *Sizning statistikangiz:*\n"
                f"• Yuklagan fayllar: {stats['current_files']}/{limits['max_files']}\n"
                f"• Qo'shgan kanallar: {stats['current_channels']}/{limits['max_channels']}\n"
                f"• Maksimal fayl hajmi: {format_size(limits['max_file_size'])}\n"
                f"• Yuklab olinganlar: {stats['files_downloaded']}\n"
                f"• Ko'rishlar: {stats['total_views']}\n\n"
                f"👇 Quyidagi tugmalardan birini tanlang:"
            )
            
            is_super = self.db.is_super_admin(user.id)
            
            await message.answer(
                welcome_text,
                parse_mode="Markdown",
                reply_markup=get_main_menu_keyboard(is_super)
            )
        
        @self.dp.callback_query(F.data == "back_to_main")
        async def back_to_main(callback: CallbackQuery):
            user_id = callback.from_user.id
            limits = self.db.get_user_limits(user_id)
            stats = self.db.get_user_stats(user_id)
            
            text = (
                f"✨ *Bosh menyu* ✨\n\n"
                f"📊 *Sizning statistikangiz:*\n"
                f"• Yuklagan fayllar: {stats['current_files']}/{limits['max_files']}\n"
                f"• Qo'shgan kanallar: {stats['current_channels']}/{limits['max_channels']}\n"
                f"• Yuklab olinganlar: {stats['files_downloaded']}"
            )
            
            is_super = self.db.is_super_admin(user_id)
            
            try:
                await callback.message.edit_text(
                    text,
                    parse_mode="Markdown",
                    reply_markup=get_main_menu_keyboard(is_super)
                )
            except:
                await callback.message.answer(
                    text,
                    parse_mode="Markdown",
                    reply_markup=get_main_menu_keyboard(is_super)
                )
            await callback.answer()
        
        @self.dp.callback_query(F.data == "cancel")
        async def cancel_action(callback: CallbackQuery, state: FSMContext):
            await state.clear()
            await callback.message.edit_text(
                "❌ *Amal bekor qilindi*",
                parse_mode="Markdown"
            )
            await back_to_main(callback)
        
        @self.dp.callback_query(F.data == "about_bot")
        async def about_bot(callback: CallbackQuery):
            text = (
                "🤖 *Bot haqida*\n\n"
                "Bu bot fayllarni maxsus kodlar orqali tarqatish uchun yaratilgan.\n\n"
                "📌 *Imkoniyatlar:*\n"
                "• Har bir foydalanuvchi o'z fayllarini yuklashi\n"
                "• O'z kanallarini qo'shishi\n"
                "• Kod orqali fayl almashish\n"
                "• Obuna tekshirish tizimi\n\n"
                "⚡️ *Limitlar:*\n"
                "• Maksimal 10 ta fayl to'plami\n"
                "• Maksimal 3 ta kanal\n"
                "• 50 MB gacha fayllar"
            )
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data == "about_creator")
        async def about_creator(callback: CallbackQuery):
            text = (
                "👨‍💻 *Yaratuvchi haqida*\n\n"
                "• *Ism:* Farrux\n"
                "• *Username:* @devc0derweb\n"
                "• *Kasb:* Python Developer\n\n"
                "📞 *Bog'lanish:* @devc0derweb"
            )
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data == "settings")
        async def settings_menu(callback: CallbackQuery):
            await callback.message.edit_text(
                "⚙️ *Sozlamalar*",
                parse_mode="Markdown",
                reply_markup=get_settings_keyboard()
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data == "view_limits")
        async def view_limits(callback: CallbackQuery):
            user_id = callback.from_user.id
            limits = self.db.get_user_limits(user_id)
            stats = self.db.get_user_stats(user_id)
            
            text = (
                "📊 *Sizning limitlaringiz*\n\n"
                f"📁 *Fayl to'plamlari:* {stats['current_files']}/{limits['max_files']}\n"
                f"📢 *Kanallar:* {stats['current_channels']}/{limits['max_channels']}\n"
                f"📦 *Maksimal fayl hajmi:* {format_size(limits['max_file_size'])}\n\n"
                f"📈 *Foydalanish:*\n"
                f"• Yuklab olingan: {stats['files_downloaded']}\n"
                f"• Ko'rishlar: {stats['total_views']}"
            )
            
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()
        
        # ========== CODE HANDLER ==========
        
        @self.dp.callback_query(F.data == "send_code")
        async def send_code_prompt(callback: CallbackQuery, state: FSMContext):
            await callback.message.edit_text(
                "🔑 *Fayl kodini yuboring*\n\n"
                "Kod 8 ta belgidan iborat (A-Z, 0-9)\n\n"
                "Masalan: `ABC12345`",
                parse_mode="Markdown",
                reply_markup=get_cancel_keyboard()
            )
            await state.set_state(UserStates.waiting_for_code)
            await callback.answer()
        
        @self.dp.message(UserStates.waiting_for_code)
        async def process_code_from_state(message: Message, state: FSMContext):
            code = message.text.strip().upper()
            
            if not CodeGenerator.validate_code(code):
                await message.answer(
                    "❌ *Noto'g'ri kod formati!*\n\n"
                    "Kod 8 ta belgidan iborat bo'lishi kerak (A-Z, 0-9).\n"
                    "Masalan: `ABC12345`",
                    parse_mode="Markdown"
                )
                return
            
            await self.handle_code(message, code, state)
        
        @self.dp.message(lambda message: message.text and len(message.text.strip()) == 8)
        async def process_code_direct(message: Message, state: FSMContext):
            code = message.text.strip().upper()
            if CodeGenerator.validate_code(code):
                await self.handle_code(message, code, state)
        
        @self.dp.callback_query(F.data.startswith("check_sub_"))
        async def check_subscription_for_code(callback: CallbackQuery, state: FSMContext):
            """Obunani tekshirish - TO'G'RILANGAN VERSIYA"""
            try:
                # Callback datadan kodni olish
                code = callback.data.split("_")[2]
                logger.info(f"Checking subscription for code: {code}")
            except Exception as e:
                logger.error(f"Error parsing callback data: {e}")
                # Agar callback datadan ololmasa, statedan olish
                data = await state.get_data()
                code = data.get('pending_code')
                if not code:
                    await callback.message.edit_text("❌ Kod topilmadi! Iltimos qaytadan urinib ko'ring.")
                    await back_to_main(callback)
                    return
            
            user_id = callback.from_user.id
            file_data = self.db.get_file_by_code(code)
            
            if not file_data:
                await callback.message.edit_text(
                    "❌ *Kod topilmadi yoki fayl o'chirilgan!*",
                    parse_mode="Markdown"
                )
                await back_to_main(callback)
                return
            
            # Fayl egasining kanallarini olish
            owner_channels = self.db.get_owner_channels(file_data['user_id'])
            
            if not owner_channels:
                # Agar kanallar bo'lmasa, to'g'ridan-to'g'ri faylni yuborish
                await self.send_file_content(callback.message, file_data, user_id)
                try:
                    await callback.message.delete()
                except:
                    pass
                await state.clear()
                await callback.answer()
                return
            
            # Obunani tekshirish
            not_subscribed, subscribed = await self.channel_checker.check_subscription(user_id, owner_channels)
            
            if not_subscribed:
                # Obuna bo'lmagan kanallar bor
                await callback.message.edit_text(
                    "❗️ *Hali ham quyidagi kanallarga obuna bo'lmagansiz:*\n\n"
                    "Barcha kanallarga obuna bo'lgach qayta urinib ko'ring.",
                    parse_mode="Markdown",
                    reply_markup=get_subscription_keyboard(not_subscribed, code)
                )
            else:
                # Barcha kanallarga obuna bo'lgan
                await self.send_file_content(callback.message, file_data, user_id)
                try:
                    await callback.message.delete()
                except:
                    pass
                await state.clear()
            
            await callback.answer()
        
        # ========== FILE UPLOAD ==========
        
        @self.dp.callback_query(F.data == "upload_file")
        async def upload_file_start(callback: CallbackQuery, state: FSMContext):
            user_id = callback.from_user.id
            limits = self.db.get_user_limits(user_id)
            current_files = self.db.get_user_files_count(user_id)
            
            if current_files >= limits['max_files'] and not self.db.is_super_admin(user_id):
                await callback.answer(
                    f"❌ Maksimal fayl soniga yetdingiz ({limits['max_files']})!",
                    show_alert=True
                )
                return
            
            await callback.message.edit_text(
                "📤 *Fayl yuklash*\n\n"
                f"Limit: {current_files}/{limits['max_files']}\n"
                f"Maksimal fayl hajmi: {format_size(limits['max_file_size'])}\n\n"
                "Fayllarni yuboring. Bir nechta fayl yuborishingiz mumkin.\n"
                "Yuklash tugagach /done buyrug'ini yuboring.",
                parse_mode="Markdown",
                reply_markup=get_cancel_keyboard()
            )
            await state.set_state(UserStates.waiting_for_files)
            await state.update_data(files=[], total_size=0)
            await callback.answer()
        
        @self.dp.message(UserStates.waiting_for_files)
        async def process_files(message: Message, state: FSMContext):
            user_id = message.from_user.id
            limits = self.db.get_user_limits(user_id)
            
            if message.text and message.text.lower() == '/done':
                data = await state.get_data()
                files = data.get('files', [])
                
                if not files:
                    await message.answer(
                        "❌ *Hech qanday fayl yuklanmadi*",
                        parse_mode="Markdown"
                    )
                    await state.clear()
                    return
                
                code = CodeGenerator.generate_code()
                file_id = self.db.create_file(code, message.from_user.id)
                
                for file_data in files:
                    self.db.add_file_item(
                        file_id=file_id,
                        file_type=file_data['type'],
                        telegram_file_id=file_data['file_id'],
                        file_name=file_data.get('name'),
                        file_size=file_data.get('size')
                    )
                
                await message.answer(
                    f"✅ *Fayllar muvaffaqiyatli yuklandi!*\n\n"
                    f"📌 *Kod:* `{code}`\n"
                    f"📊 *Fayllar soni:* {len(files)}\n"
                    f"📦 *Umumiy hajm:* {format_size(data['total_size'])}\n\n"
                    f"🔍 Bu kodni boshqalarga yuboring!",
                    parse_mode="Markdown"
                )
                await state.clear()
                return
            
            file_size = 0
            file_data = None
            
            if message.photo:
                file_data = {
                    'type': 'photo',
                    'file_id': message.photo[-1].file_id,
                    'name': f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    'size': None
                }
            elif message.video:
                file_size = message.video.file_size or 0
                if file_size > limits['max_file_size'] and not self.db.is_super_admin(user_id):
                    await message.answer(
                        f"❌ *Fayl hajmi juda katta!*\n\n"
                        f"Sizning limitingiz: {format_size(limits['max_file_size'])}\n"
                        f"Yuborgan fayl: {format_size(file_size)}",
                        parse_mode="Markdown"
                    )
                    return
                
                file_data = {
                    'type': 'video',
                    'file_id': message.video.file_id,
                    'name': message.video.file_name or f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    'size': file_size
                }
            elif message.document:
                file_size = message.document.file_size or 0
                if file_size > limits['max_file_size'] and not self.db.is_super_admin(user_id):
                    await message.answer(
                        f"❌ *Fayl hajmi juda katta!*\n\n"
                        f"Sizning limitingiz: {format_size(limits['max_file_size'])}\n"
                        f"Yuborgan fayl: {format_size(file_size)}",
                        parse_mode="Markdown"
                    )
                    return
                
                file_data = {
                    'type': 'document',
                    'file_id': message.document.file_id,
                    'name': message.document.file_name or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'size': file_size
                }
            
            if file_data:
                data = await state.get_data()
                files = data.get('files', [])
                total_size = data.get('total_size', 0) + (file_size or 0)
                
                files.append(file_data)
                await state.update_data(files=files, total_size=total_size)
                
                size_info = f" | {format_size(file_data['size'])}" if file_data['size'] else ""
                await message.answer(
                    f"✅ *Fayl qabul qilindi*\n\n"
                    f"📁 {file_data['name']}{size_info}\n"
                    f"📊 Jami: {len(files)} ta fayl | {format_size(total_size)}\n\n"
                    f"Yana fayl yuborishingiz yoki /done yozishingiz mumkin.",
                    parse_mode="Markdown"
                )
            else:
                await message.answer(
                    "❌ *Iltimos, fayl yuboring*\n\n"
                    "Rasm, video yoki hujjat yuborishingiz mumkin.",
                    parse_mode="Markdown"
                )
        
        # ========== MY FILES ==========
        
        @self.dp.callback_query(F.data == "my_files")
        async def my_files(callback: CallbackQuery):
            user_id = callback.from_user.id
            files = self.db.get_user_files(user_id)
            
            if not files:
                await callback.message.edit_text(
                    "📂 *Sizda hali fayllar mavjud emas*",
                    parse_mode="Markdown",
                    reply_markup=get_back_keyboard()
                )
                await callback.answer()
                return
            
            text = f"📂 *Sizning fayllaringiz*\n\nJami: {len(files)} ta to'plam"
            
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_files_keyboard(files)
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data.startswith("view_file_"))
        async def view_file(callback: CallbackQuery):
            file_id = int(callback.data.split("_")[2])
            user_id = callback.from_user.id
            
            file_data = None
            files = self.db.get_user_files(user_id)
            for f in files:
                if f['id'] == file_id:
                    file_data = f
                    break
            
            is_owner = file_data is not None
            
            await callback.message.edit_text(
                f"📁 *Fayl amallari*\n\nKod: `{file_data['code'] if file_data else 'Noma\'lum'}`",
                parse_mode="Markdown",
                reply_markup=get_file_actions_keyboard(file_id, is_owner)
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data.startswith("download_file_"))
        async def download_file(callback: CallbackQuery):
            file_id = int(callback.data.split("_")[2])
            user_id = callback.from_user.id
            file_items = self.db.get_file_items(file_id)
            
            self.db.log_download(file_id, user_id)
            
            await callback.message.answer(
                f"📥 *{len(file_items)} ta fayl yuborilmoqda...*",
                parse_mode="Markdown"
            )
            
            for item in file_items:
                try:
                    caption = f"📁 {item['file_name'] or 'Fayl'}"
                    if item['file_size']:
                        caption += f" | {format_size(item['file_size'])}"
                    
                    if item['file_type'] == 'photo':
                        await callback.message.answer_photo(
                            item['telegram_file_id'],
                            caption=caption
                        )
                    elif item['file_type'] == 'video':
                        await callback.message.answer_video(
                            item['telegram_file_id'],
                            caption=caption
                        )
                    else:
                        await callback.message.answer_document(
                            item['telegram_file_id'],
                            caption=caption
                        )
                except Exception as e:
                    logger.error(f"Error sending file: {e}")
                    try:
                        await callback.message.answer_document(item['telegram_file_id'])
                    except:
                        pass
            
            await callback.answer()
        
        @self.dp.callback_query(F.data.startswith("delete_file_"))
        async def delete_file(callback: CallbackQuery):
            file_id = int(callback.data.split("_")[2])
            user_id = callback.from_user.id
            
            if self.db.delete_file(file_id, user_id):
                await callback.message.edit_text(
                    "✅ *Fayl muvaffaqiyatli o'chirildi!*",
                    parse_mode="Markdown"
                )
            else:
                await callback.message.edit_text(
                    "❌ *Faylni o'chirishda xatolik!*\n\n"
                    "Siz faqat o'zingizning fayllaringizni o'chira olasiz.",
                    parse_mode="Markdown"
                )
            
            await back_to_main(callback)
        
        # ========== CHANNEL MANAGEMENT ==========
        
        @self.dp.callback_query(F.data == "add_channel")
        async def add_channel_start(callback: CallbackQuery, state: FSMContext):
            user_id = callback.from_user.id
            limits = self.db.get_user_limits(user_id)
            current_channels = self.db.get_user_channels_count(user_id)
            
            if current_channels >= limits['max_channels'] and not self.db.is_super_admin(user_id):
                await callback.answer(
                    f"❌ Maksimal kanal soniga yetdingiz ({limits['max_channels']})!",
                    show_alert=True
                )
                return
            
            await callback.message.edit_text(
                "➕ *Kanal qo'shish*\n\n"
                f"Limit: {current_channels}/{limits['max_channels']}\n\n"
                "Kanal username yoki linkini yuboring:\n"
                "Masalan: `@kanal_nomi` yoki `https://t.me/kanal_nomi`\n\n"
                "⚠️ Bot kanalda admin bo'lishi kerak!",
                parse_mode="Markdown",
                reply_markup=get_cancel_keyboard()
            )
            await state.set_state(UserStates.waiting_for_channel)
            await callback.answer()
        
        @self.dp.message(UserStates.waiting_for_channel)
        async def process_channel(message: Message, state: FSMContext):
            user_id = message.from_user.id
            channel_username = extract_channel_username(message.text)
            
            if not channel_username:
                await message.answer(
                    "❌ *Noto'g'ri kanal formati*",
                    parse_mode="Markdown"
                )
                return
            
            channel_info = await self.channel_checker.get_channel_info(channel_username)
            
            if not channel_info:
                await message.answer(
                    "❌ *Kanal topilmadi yoki bot kanalda admin emas*",
                    parse_mode="Markdown"
                )
                return
            
            if self.db.add_channel(
                channel_id=channel_info['id'],
                channel_title=channel_info['title'],
                user_id=user_id,
                channel_username=channel_info['username']
            ):
                await message.answer(
                    f"✅ *Kanal muvaffaqiyatli qo'shildi!*\n\n"
                    f"📢 {channel_info['title']}",
                    parse_mode="Markdown"
                )
            else:
                await message.answer(
                    "❌ *Kanal qo'shishda xatolik*",
                    parse_mode="Markdown"
                )
            
            await state.clear()
        
        @self.dp.callback_query(F.data == "remove_channel")
        async def remove_channel_list(callback: CallbackQuery):
            user_id = callback.from_user.id
            channels = self.db.get_user_channels(user_id)
            
            if not channels:
                await callback.message.edit_text(
                    "❌ *Sizda kanallar mavjud emas*",
                    parse_mode="Markdown",
                    reply_markup=get_back_keyboard()
                )
                await callback.answer()
                return
            
            await callback.message.edit_text(
                "➖ *O'chiriladigan kanalni tanlang*",
                parse_mode="Markdown",
                reply_markup=get_channels_keyboard(channels)
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data.startswith("view_channel_"))
        async def view_channel(callback: CallbackQuery):
            channel_id = int(callback.data.split("_")[2])
            
            await callback.message.edit_text(
                "📢 *Kanal amallari*",
                parse_mode="Markdown",
                reply_markup=get_channel_actions_keyboard(channel_id)
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data.startswith("remove_my_channel_"))
        async def remove_my_channel(callback: CallbackQuery):
            channel_id = int(callback.data.split("_")[3])
            user_id = callback.from_user.id
            
            if self.db.remove_channel(channel_id, user_id):
                await callback.message.edit_text(
                    "✅ *Kanal muvaffaqiyatli o'chirildi!*",
                    parse_mode="Markdown"
                )
            else:
                await callback.message.edit_text(
                    "❌ *Kanal o'chirishda xatolik*",
                    parse_mode="Markdown"
                )
            
            await back_to_main(callback)
        
        @self.dp.callback_query(F.data == "list_my_channels")
        async def list_my_channels(callback: CallbackQuery):
            user_id = callback.from_user.id
            channels = self.db.get_user_channels(user_id)
            
            if not channels:
                await callback.message.edit_text(
                    "📋 *Sizda kanallar mavjud emas*",
                    parse_mode="Markdown",
                    reply_markup=get_back_keyboard()
                )
                await callback.answer()
                return
            
            text = "📋 *Sizning kanallaringiz:*\n\n"
            for i, channel in enumerate(channels, 1):
                username = f"@{channel['channel_username']}" if channel['channel_username'] else "Maxfiy kanal"
                text += f"{i}. *{channel['channel_title']}*\n   {username}\n"
            
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()
        
        # ========== STATISTICS ==========
        
        @self.dp.callback_query(F.data == "my_stats")
        async def my_stats(callback: CallbackQuery):
            user_id = callback.from_user.id
            stats = self.db.get_user_stats(user_id)
            limits = self.db.get_user_limits(user_id)
            
            text = (
                "📊 *Mening statistikam*\n\n"
                f"📁 *Fayllar:* {stats['current_files']}/{limits['max_files']}\n"
                f"📤 *Yuklangan:* {stats['files_uploaded']}\n"
                f"📥 *Yuklab olingan:* {stats['files_downloaded']}\n"
                f"👁 *Ko'rishlar:* {stats['total_views']}\n"
                f"📢 *Kanallar:* {stats['current_channels']}/{limits['max_channels']}\n\n"
                f"📦 *Max fayl hajmi:* {format_size(limits['max_file_size'])}"
            )
            
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()
        
        # ========== SUPER ADMIN HANDLERS ==========
        
        @self.dp.callback_query(F.data == "super_admin")
        async def super_admin_panel(callback: CallbackQuery):
            if not self.db.is_super_admin(callback.from_user.id):
                await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
                return
            
            await callback.message.edit_text(
                "👑 *Super Admin Panel*",
                parse_mode="Markdown",
                reply_markup=get_super_admin_keyboard()
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data == "sa_users")
        async def sa_users(callback: CallbackQuery):
            if not self.db.is_super_admin(callback.from_user.id):
                await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
                return
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) as total,
                           SUM(is_admin) as admins,
                           SUM(is_super_admin) as super_admins
                    FROM users
                ''')
                row = cursor.fetchone()
                stats = dict(row) if row else {'total': 0, 'admins': 0, 'super_admins': 0}
            
            text = (
                "👥 *Foydalanuvchilar statistikasi*\n\n"
                f"• Jami: {stats['total']}\n"
                f"• Adminlar: {stats['admins']}\n"
                f"• Super adminlar: {stats['super_admins']}"
            )
            
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data == "sa_stats")
        async def sa_stats(callback: CallbackQuery):
            if not self.db.is_super_admin(callback.from_user.id):
                await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
                return
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM files WHERE is_active = 1')
                total_files = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM file_items')
                total_items = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM channels WHERE is_active = 1')
                total_channels = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM downloads')
                total_downloads = cursor.fetchone()[0]
                
                cursor.execute('SELECT SUM(views) FROM files')
                total_views = cursor.fetchone()[0] or 0
            
            text = (
                "📊 *Umumiy statistika*\n\n"
                f"📁 Fayllar: {total_files}\n"
                f"📄 Fayl elementlari: {total_items}\n"
                f"📢 Kanallar: {total_channels}\n"
                f"📥 Yuklamalar: {total_downloads}\n"
                f"👁 Ko'rishlar: {total_views}"
            )
            
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()
        
        @self.dp.callback_query(F.data == "sa_all_channels")
        async def sa_all_channels(callback: CallbackQuery):
            if not self.db.is_super_admin(callback.from_user.id):
                await callback.answer("❌ Ruxsat yo'q!", show_alert=True)
                return
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT c.*, u.telegram_id, u.username, u.first_name
                    FROM channels c
                    JOIN users u ON c.user_id = u.telegram_id
                    WHERE c.is_active = 1
                    ORDER BY c.added_at DESC
                ''')
                channels = [dict(row) for row in cursor.fetchall()]
            
            if not channels:
                await callback.message.edit_text(
                    "📋 *Kanallar mavjud emas*",
                    parse_mode="Markdown",
                    reply_markup=get_back_keyboard()
                )
                await callback.answer()
                return
            
            text = "📋 *Barcha kanallar:*\n\n"
            for i, channel in enumerate(channels, 1):
                owner_name = f"@{channel['username']}" if channel['username'] else channel['first_name']
                username = f"@{channel['channel_username']}" if channel['channel_username'] else "Maxfiy"
                
                text += f"{i}. *{channel['channel_title']}*\n   👤 {owner_name}\n   🔗 {username}\n\n"
            
            # Agar text juda uzun bo'lsa, qisqartirish
            if len(text) > 4000:
                text = text[:4000] + "...\n\n(Juda ko'p kanallar)"
            
            await callback.message.edit_text(
                text,
                parse_mode="Markdown",
                reply_markup=get_back_keyboard()
            )
            await callback.answer()

# ==================== MAIN BOT CLASS ====================

class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=BOT_TOKEN)
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        self.db = Database()
        self.channel_checker = ChannelChecker(self.bot)
        
        # Setup middleware
        self.dp.message.middleware(SubscriptionMiddleware(self.db, self.channel_checker))
        self.dp.callback_query.middleware(SubscriptionMiddleware(self.db, self.channel_checker))
        
        # Setup handlers
        self.handlers = BotHandlers(self.bot, self.dp, self.db, self.channel_checker)
    
    async def start(self):
        try:
            logger.info("Bot started successfully")
            print("🤖 Bot ishga tushdi! Multi-user mode enabled")
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            print(f"❌ Bot ishdan chiqdi: {e}")
        finally:
            await self.bot.session.close()
    
    def run(self):
        asyncio.run(self.start())

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not found in environment variables!")
        print("❌ BOT_TOKEN topilmadi! .env faylini tekshiring.")
        print("\n.env faylini quyidagicha yarating:")
        print("BOT_TOKEN=7532713270:AAH7YwYh_7CVmhM0a3Iu0rVqO3HKYgBw1_o")
        print("ADMIN_IDS=123456789,987654321")
        print("DATABASE_PATH=bot_database.db")
        exit(1)
    
    print("""
    ╔════════════════════════════════════╗
    ║     🤖 Telegram Bot v2.0           ║
    ║     📁 Multi-user File Sharing     ║
    ║     👨‍💻 Yaratuvchi: @devc0derweb   ║
    ╚════════════════════════════════════╝
    """)
    
    bot = TelegramBot()
    bot.run()