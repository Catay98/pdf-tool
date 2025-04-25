"""
配置文件，加载环境变量并设置应用配置
"""

import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量（如果存在）
load_dotenv()

class Config:
    """基础配置类"""
    # 密钥，用于会话加密
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    # 上传文件配置
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER') or 'results'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # 其他配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    
    # 在生产环境中，应使用更安全的设置
    # 确保SECRET_KEY在生产环境中被设置
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # 日志处理
        import logging
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        # 确保HTTPS
        from flask_talisman import Talisman
        Talisman(app)

# 环境配置字典
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# 当前环境
env = os.environ.get('FLASK_ENV') or 'default'
current_config = config[env]