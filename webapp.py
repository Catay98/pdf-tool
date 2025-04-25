#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF知识点提炼工具 - Web应用界面
提供基于Flask的Web界面，允许用户上传PDF并获取知识点提炼结果
"""

import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session
from flask_wtf.csrf import CSRFProtect

# 导入配置
from config import current_config

# 导入主程序模块
from main import PDFKnowledgeTool

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(current_config)
csrf = CSRFProtect(app)

# 配置日志
log_level = getattr(logging, app.config['LOG_LEVEL'])
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/webapp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保上传和输出目录存在
upload_folder = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
output_folder = os.path.join(os.getcwd(), app.config['OUTPUT_FOLDER'])

os.makedirs(upload_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 更新配置中的路径为绝对路径
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['OUTPUT_FOLDER'] = output_folder

logger.info(f"上传目录: {app.config['UPLOAD_FOLDER']}")
logger.info(f"结果目录: {app.config['OUTPUT_FOLDER']}")

# 创建PDF知识点工具实例
pdf_tool = PDFKnowledgeTool()

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    # 检查是否有文件
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(request.url)
    
    file = request.files['file']
    
    # 如果用户没有选择文件
    if file.filename == '':
        flash('没有选择文件')
        return redirect(request.url)
    
    # 检查文件类型和处理上传
    if file and allowed_file(file.filename):
        try:
            # 安全地获取文件名并保存
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            logger.info(f"文件已上传: {file_path}")
            
            # 获取表单参数
            output_format = request.form.get('format', 'markdown')
            extract_mode = request.form.get('mode', 'auto')
            min_importance = float(request.form.get('importance', 0.5))
            
            # 处理PDF
            output_path = pdf_tool.process_pdf(
                file_path,
                output_format=output_format,
                extract_mode=extract_mode,
                min_importance=min_importance,
                output_dir=app.config['OUTPUT_FOLDER']
            )
            
            # 保存结果路径到会话
            session['result_path'] = output_path
            session['original_filename'] = filename
            
            # 重定向到结果页面
            return redirect(url_for('show_result'))
            
        except Exception as e:
            logger.error(f"处理上传文件时出错: {str(e)}", exc_info=True)
            flash(f'处理文件时发生错误: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('只允许上传PDF文件')
        return redirect(url_for('index'))

@app.route('/result')
def show_result():
    """显示处理结果"""
    # 检查是否有结果路径
    if 'result_path' not in session:
        flash('没有可用的处理结果')
        return redirect(url_for('index'))
    
    result_path = session['result_path']
    original_filename = session.get('original_filename', '未知文件')
    
    # 检查结果文件是否存在
    if not os.path.exists(result_path):
        flash('结果文件不存在')
        return redirect(url_for('index'))
    
    # 读取结果文件内容
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确定文件格式
        format_type = os.path.splitext(result_path)[1].lstrip('.')
        
        return render_template(
            'result.html',
            content=content,
            format=format_type,
            filename=original_filename,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        logger.error(f"读取结果文件时出错: {str(e)}", exc_info=True)
        flash(f'读取结果时发生错误: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download')
def download_result():
    """下载处理结果"""
    # 检查是否有结果路径
    if 'result_path' not in session:
        flash('没有可用的处理结果')
        return redirect(url_for('index'))
    
    result_path = session['result_path']
    original_filename = session.get('original_filename', '未知文件')
    
    # 检查结果文件是否存在
    if not os.path.exists(result_path):
        flash('结果文件不存在')
        return redirect(url_for('index'))
    
    # 确定下载的文件名
    format_type = os.path.splitext(result_path)[1]
    download_name = f"{os.path.splitext(original_filename)[0]}_知识点{format_type}"
    
    return send_file(
        result_path,
        as_attachment=True,
        download_name=download_name
    )

@app.route('/batch', methods=['GET', 'POST'])
def batch_process():
    """批量处理页面和功能"""
    if request.method == 'POST':
        # 检查是否有文件
        if 'files[]' not in request.files:
            flash('没有选择文件')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        
        # 如果没有选择文件
        if not files or files[0].filename == '':
            flash('没有选择文件')
            return redirect(request.url)
        
        # 获取表单参数
        output_format = request.form.get('format', 'markdown')
        extract_mode = request.form.get('mode', 'auto')
        min_importance = float(request.form.get('importance', 0.5))
        
        # 处理结果信息
        results = []
        errors = []
        
        # 处理每个文件
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # 安全地获取文件名并保存
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{timestamp}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(file_path)
                    
                    # 处理PDF
                    output_path = pdf_tool.process_pdf(
                        file_path,
                        output_format=output_format,
                        extract_mode=extract_mode,
                        min_importance=min_importance,
                        output_dir=app.config['OUTPUT_FOLDER']
                    )
                    
                    results.append({
                        'original_filename': filename,
                        'result_path': output_path
                    })
                    
                except Exception as e:
                    logger.error(f"处理文件 {file.filename} 时出错: {str(e)}", exc_info=True)
                    errors.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
            else:
                errors.append({
                    'filename': file.filename,
                    'error': '不支持的文件类型'
                })
        
        # 保存结果到会话
        session['batch_results'] = results
        session['batch_errors'] = errors
        
        return redirect(url_for('batch_results'))
    
    # GET请求，显示批量处理表单
    return render_template('batch.html')

@app.route('/batch/results')
def batch_results():
    """显示批量处理结果"""
    # 检查是否有结果
    if 'batch_results' not in session:
        flash('没有可用的批量处理结果')
        return redirect(url_for('batch_process'))
    
    results = session['batch_results']
    errors = session.get('batch_errors', [])
    
    return render_template(
        'batch_results.html',
        results=results,
        errors=errors,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route('/download/<path:filename>')
def download_file(filename):
    """下载指定文件"""
    # 安全检查，确保文件在OUTPUT_FOLDER内
    full_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(filename))
    
    if not os.path.exists(full_path):
        flash('文件不存在')
        return redirect(url_for('index'))
    
    return send_file(
        full_path,
        as_attachment=True
    )

@app.errorhandler(413)
def request_entity_too_large(error):
    """处理文件过大错误"""
    flash('上传的文件太大，请确保文件小于16MB')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    """处理内部服务器错误"""
    logger.error(f"内部服务器错误: {str(error)}", exc_info=True)
    flash('服务器内部错误，请稍后再试')
    return redirect(url_for('index'))

# HTML模板目录默认为"templates"文件夹
# 创建必要的HTML模板:
# - templates/index.html (上传表单)
# - templates/result.html (结果展示)
# - templates/batch.html (批量处理表单)
# - templates/batch_results.html (批量处理结果)
# - templates/layout.html (基础布局模板)

if __name__ == '__main__':
    # 从环境变量获取端口或默认使用5000
    port = int(os.environ.get('PORT', 5000))
    # 从环境变量获取绑定地址或默认使用0.0.0.0（所有网络接口）
    host = os.environ.get('HOST', '0.0.0.0')
    # 从环境变量获取是否开启调试模式
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"启动Web应用在 {host}:{port}, 调试模式: {debug}")
    app.run(host=host, port=port, debug=debug)