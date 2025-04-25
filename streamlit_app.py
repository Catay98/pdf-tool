import streamlit as st
import io
import os
import re
import base64
import tempfile
from PIL import Image
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# 初始化NLTK资源（首次运行时下载）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 安装并导入必要的PDF处理库
import sys
try:
    import PyPDF2
except ImportError:
    os.system(f"{sys.executable} -m pip install PyPDF2==3.0.1")
    import PyPDF2

try:
    import pdfplumber
except ImportError:
    os.system(f"{sys.executable} -m pip install pdfplumber")
    import pdfplumber

try:
    import pytesseract
    from pdf2image import convert_from_bytes, convert_from_path
except ImportError:
    os.system(f"{sys.executable} -m pip install pytesseract pdf2image")
    import pytesseract
    from pdf2image import convert_from_bytes, convert_from_path

# 页面配置
st.set_page_config(
    page_title="PDF知识点提炼工具",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .knowledge-point {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .highlight {
        background-color: #fff3cd;
        font-weight: bold;
    }
    .debug-info {
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
        max-height: 200px;
        overflow-y: auto;
    }
    .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .warning-message {
        color: #ff6f00;
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def get_tessdata_path():
    """尝试定位tesseract数据目录"""
    # 检查是否在Streamlit Cloud环境
    if os.path.exists('/mount/src'):
        # 尝试下载中文和英文语言数据包
        os.system("mkdir -p /tmp/tessdata")
        os.system("curl -L 'https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata' -o /tmp/tessdata/chi_sim.traineddata")
        os.system("curl -L 'https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata' -o /tmp/tessdata/eng.traineddata")
        return "/tmp/tessdata"
    return None

def extract_text_with_ocr(pdf_file, pages=None, lang='chi_sim+eng'):
    """使用OCR从PDF提取文本"""
    try:
        # 设置tessdata路径
        tessdata_path = get_tessdata_path()
        if tessdata_path:
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'
            os.environ['TESSDATA_PREFIX'] = tessdata_path
            
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_pdf_path = temp_pdf.name
            
        pdf_file.seek(0)  # 重置文件指针
        
        # 转换PDF为图像
        try:
            # 首先尝试使用临时文件
            images = convert_from_path(temp_pdf_path, dpi=300, fmt='ppm', 
                                     thread_count=4, paths_only=False)
        except Exception as e:
            # 如果失败，尝试使用内存中的文件
            pdf_file.seek(0)
            images = convert_from_bytes(pdf_file.read(), dpi=300, fmt='ppm', 
                                      thread_count=4)
        
        # 清理临时文件
        try:
            os.unlink(temp_pdf_path)
        except:
            pass
        
        # 提取特定页面或所有页面
        if pages:
            selected_images = [images[i-1] for i in pages if 0 < i <= len(images)]
        else:
            selected_images = images
            
        # 使用OCR提取文本
        text = ""
        for i, img in enumerate(selected_images):
            # 使用pytesseract执行OCR
            page_text = pytesseract.image_to_string(img, lang=lang)
            text += f"\n\n--- 第 {i+1} 页 ---\n\n" + page_text
            
        return text
        
    except Exception as e:
        st.error(f"OCR处理错误: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file, use_ocr=False, debug=False):
    """从PDF文件提取文本，支持多种方法"""
    results = {}
    errors = []
    text = ""
    
    # 方法1: 使用PyPDF2
    try:
        pdf_file.seek(0)  # 重置文件指针
        reader = PyPDF2.PdfReader(pdf_file)
        pypdf2_text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pypdf2_text += page_text + "\n\n"
        
        if pypdf2_text.strip():
            text = pypdf2_text
            results['PyPDF2'] = pypdf2_text
        else:
            errors.append("PyPDF2未能提取任何文本")
    except Exception as e:
        errors.append(f"PyPDF2错误: {str(e)}")
    
    # 方法2: 使用pdfplumber
    if not text.strip():
        try:
            pdf_file.seek(0)  # 重置文件指针
            with pdfplumber.open(pdf_file) as pdf:
                plumber_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    plumber_text += page_text + "\n\n"
                
                if plumber_text.strip():
                    text = plumber_text
                    results['pdfplumber'] = plumber_text
                else:
                    errors.append("pdfplumber未能提取任何文本")
        except Exception as e:
            errors.append(f"pdfplumber错误: {str(e)}")
    
    # 方法3: 如果前两种方法都失败或选择了OCR，使用OCR
    if (not text.strip() or use_ocr) and 'pytesseract' in sys.modules:
        try:
            pdf_file.seek(0)  # 重置文件指针
            ocr_text = extract_text_with_ocr(pdf_file)
            if ocr_text and ocr_text.strip():
                text = ocr_text
                results['OCR'] = ocr_text
            else:
                errors.append("OCR未能提取任何文本")
        except Exception as e:
            errors.append(f"OCR错误: {str(e)}")
    
    # 如果所有方法都失败
    if not text.strip():
        text = "无法提取文本。这可能是扫描版PDF没有文本层，或文件受到保护。"
    
    # 处理一些常见的PDF提取问题
    if text.strip():
        # 删除重复的行
        lines = text.splitlines()
        cleaned_lines = []
        prev_line = ""
        for line in lines:
            if line != prev_line:
                cleaned_lines.append(line)
            prev_line = line
        text = "\n".join(cleaned_lines)
        
        # 替换一些常见的乱码字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # 如果提取的文本是URL或乱码模式，尝试OCR
        if re.search(r'(https?:\/\/[^\s]+){3,}', text) and 'OCR' not in results and 'pytesseract' in sys.modules:
            try:
                pdf_file.seek(0)
                ocr_text = extract_text_with_ocr(pdf_file)
                if ocr_text and ocr_text.strip() and len(ocr_text) > len(text):
                    text = ocr_text
                    results['OCR'] = ocr_text
            except Exception as e:
                errors.append(f"尝试OCR修复错误: {str(e)}")
    
    if debug:
        return text, results, errors
    return text

def preprocess_text(text):
    """预处理文本"""
    # 移除特殊字符和多余空格
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\?\!，。；：？！、]', '', text)
    return text

def extract_sentences(text, min_length=5):
    """提取文本中的句子"""
    # 处理中文和英文混合句子
    text = re.sub(r'([。！？；!?;])', r'\1\n', text)
    sentences = []
    for line in text.split('\n'):
        if line.strip():
            # 对于英文和混合文本使用NLTK
            sentences.extend(sent_tokenize(line))
    
    # 过滤太短的句子
    return [s.strip() for s in sentences if len(s.strip().split()) >= min_length or len(s.strip()) >= 10]

def score_importance(sentence, keywords=None, stop_words=None, language='auto'):
    """评估句子的重要性"""
    if language == 'auto':
        # 简单检测语言
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sentence))
        if chinese_chars > len(sentence) / 3:
            language = 'chinese'
        else:
            language = 'english'
    
    if stop_words is None:
        try:
            if language == 'chinese':
                # 中文停用词
                stop_words = set([
                    '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '有', '在',
                    '中', '为', '对', '也', '以', '于', '上', '下', '之', '由', '等', '被'
                ])
            else:
                stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
    
    # 根据语言选择分词方法
    if language == 'chinese':
        words = list(jieba.cut(sentence)) if 'jieba' in sys.modules else list(sentence)
    else:
        words = word_tokenize(sentence.lower())
    
    # 过滤停用词
    filtered_words = [w for w in words if w not in stop_words and (w.isalnum() or re.match(r'[\u4e00-\u9fff]', w))]
    
    # 基本分数计算
    score = 0.5  # 基础分数
    
    # 句子长度因素
    length = len(filtered_words)
    if length > 20:
        score += 0.1
    elif length > 10:
        score += 0.05
    
    # 关键词匹配
    if keywords:
        matches = sum(1 for word in filtered_words if word in keywords)
        score += 0.05 * matches
    
    # 包含数字通常表示更具体的信息
    if any(c.isdigit() for c in sentence):
        score += 0.05
    
    # 特殊标记词语，通常表示重要内容
    importance_markers = {
        'english': ["important", "key", "significant", "essential", "crucial", "critical", "note", "remember"],
        'chinese': ["重要", "关键", "显著", "本质", "至关重要", "关键", "注意", "记住", "核心", "主要"]
    }
    
    markers = importance_markers.get(language, [])
    if any(marker in (words if language == 'chinese' else [w.lower() for w in words]) for marker in markers):
        score += 0.1
    
    # 句子位置特征（如果有上下文信息）
    # 这里简化处理，实际中可能需要段落级信息
    
    return min(score, 1.0)  # 确保分数不超过1.0

def extract_keywords(text, top_n=20, language='auto'):
    """提取文本中的关键词"""
    if language == 'auto':
        # 简单检测语言
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars > len(text) / 3:
            language = 'chinese'
        else:
            language = 'english'
    
    try:
        if language == 'chinese':
            # 尝试使用jieba提取中文关键词
            try:
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(text, topK=top_n)
                return keywords
            except:
                # 简单按词频提取
                words = list(text)
                stop_words = set([
                    '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '有', '在',
                    '中', '为', '对', '也', '以', '于', '上', '下', '之', '由', '等', '被'
                ])
        else:
            # 英文关键词提取
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            
        # 过滤停用词和短词
        filtered_words = [w for w in words if w not in stop_words and ((len(w) > 1 and language == 'chinese') or (len(w) > 2 and language != 'chinese'))]
        
        # 统计词频
        word_counts = Counter(filtered_words)
        
        # 返回最常见的词
        return [word for word, _ in word_counts.most_common(top_n)]
    except Exception as e:
        st.warning(f"提取关键词时出错: {str(e)}，将使用简化方法")
        # 简化方法：按词频提取
        words = text.split() if language != 'chinese' else list(text)
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_n)]

def get_chapter_headings(text):
    """尝试提取章节标题"""
    # 匹配常见的章节标题模式
    patterns = [
        # 英文模式
        r'(?:Chapter|CHAPTER)\s+\d+[:\.\s]+(.+?)(?:\n|\r\n?)',
        r'(?:Section|SECTION)\s+\d+(?:\.\d+)*[:\.\s]+(.+?)(?:\n|\r\n?)',
        r'^\d+(?:\.\d+)*[:\.\s]+(.+?)(?:\n|\r\n?)',
        # 中文模式
        r'第\s*[一二三四五六七八九十\d]+\s*[章节]\s*[：:]\s*(.+?)(?:\n|\r\n?)',
        r'[一二三四五六七八九十\d]+[\.、]\s*(.+?)(?:\n|\r\n?)',
        r'[（\(][一二三四五六七八九十\d]+[）\)]\s*(.+?)(?:\n|\r\n?)'
    ]
    
    headings = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        headings.extend(matches)
    
    # 去重并按原始顺序保留
    seen = set()
    unique_headings = []
    for h in headings:
        h_stripped = h.strip()
        if h_stripped and h_stripped not in seen and len(h_stripped) < 100:  # 避免匹配过长的内容
            seen.add(h_stripped)
            unique_headings.append(h_stripped)
    
    return unique_headings

def create_downloadable_link(content, filename, link_text):
    """创建可下载内容的链接"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="btn btn-primary">{link_text}</a>'
    return href

def extract_knowledge_points(text, mode, importance_threshold):
    """根据不同模式提取知识点"""
    text = preprocess_text(text)
    
    # 检测语言
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    language = 'chinese' if chinese_chars > len(text) / 10 else 'english'
    
    if mode == "自动模式":
        # 检测文档结构，选择适合的模式
        headings = get_chapter_headings(text)
        if len(headings) > 2:
            mode = "章节模式"
        else:
            mode = "句子模式"
    
    # 提取关键词
    keywords = extract_keywords(text, language=language)
    
    if mode == "关键词模式":
        # 直接返回关键词作为知识点
        return [{"text": kw, "importance": 0.7, "type": "keyword"} for kw in keywords if kw]
    
    elif mode == "句子模式":
        # 提取和评分句子
        sentences = extract_sentences(text)
        knowledge_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            importance = score_importance(sentence, keywords, language=language)
            if importance >= importance_threshold:
                knowledge_points.append({
                    "text": sentence,
                    "importance": importance,
                    "type": "sentence"
                })
        
        # 按重要性降序排序
        return sorted(knowledge_points, key=lambda x: x["importance"], reverse=True)
    
    elif mode == "章节模式":
        # 提取章节标题和相关内容
        headings = get_chapter_headings(text)
        
        if not headings:
            # 如果没有找到章节标题，回退到句子模式
            st.warning("未找到章节标题，已自动切换到句子模式")
            return extract_knowledge_points(text, "句子模式", importance_threshold)
        
        sections = []
        
        # 将文本按章节拆分
        text_parts = []
        current_pos = 0
        
        for i, heading in enumerate(headings):
            # 查找当前标题
            heading_pos = text.find(heading, current_pos)
            
            if heading_pos == -1:
                continue
                
            # 如果不是第一个标题，将前一个标题到当前标题之间的内容作为前一个标题的内容
            if i > 0 and heading_pos > current_pos:
                text_parts.append(text[current_pos:heading_pos])
                
            current_pos = heading_pos + len(heading)
            
            # 如果是最后一个标题，将剩余内容作为最后一个标题的内容
            if i == len(headings) - 1:
                text_parts.append(text[current_pos:])
                
        # 如果找到的章节数与标题数不匹配，使用简单的分块方法
        if len(text_parts) != len(headings):
            # 简单分块
            chunks = re.split(r'(第\s*[一二三四五六七八九十\d]+\s*[章节]|Chapter\s+\d+|Section\s+\d+(?:\.\d+)*)', text)
            # 重建文本部分
            text_parts = []
            for i in range(1, len(chunks), 2):
                if i+1 < len(chunks):
                    text_parts.append(chunks[i+1])
                
            # 如果还是不匹配，使用平均分块
            if len(text_parts) != len(headings):
                text_parts = []
                chunk_size = len(text) // len(headings)
                for i in range(len(headings)):
                    start = i * chunk_size
                    end = start + chunk_size if i < len(headings) - 1 else len(text)
                    text_parts.append(text[start:end])
        
        # 处理每个章节
        for i, heading in enumerate(headings):
            if i < len(text_parts):
                section_text = text_parts[i]
                # 为每个章节提取关键句子
                sentences = extract_sentences(section_text)
                section_points = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    importance = score_importance(sentence, keywords, language=language)
                    if importance >= importance_threshold:
                        section_points.append({
                            "text": sentence,
                            "importance": importance,
                            "type": "sentence"
                        })
                
                # 按重要性降序排序
                section_points = sorted(section_points, key=lambda x: x["importance"], reverse=True)
                max_points = min(10, len(section_points))  # 每章最多10个要点
                
                sections.append({
                    "heading": heading,
                    "importance": 0.9,  # 章节标题通常很重要
                    "type": "heading",
                    "points": section_points[:max_points]
                })
        
        return sections

def format_knowledge_points(knowledge_points, mode, output_format):
    """格式化知识点为指定格式"""
    if mode == "章节模式":
        if output_format == "Markdown":
            content = "# PDF文档知识点提炼\n\n"
            
            for section in knowledge_points:
                content += f"## {section['heading']}\n\n"
                
                for point in section['points']:
                    content += f"- {point['text']}\n\n"
            
        else:  # 纯文本
            content = "PDF文档知识点提炼\n\n"
            
            for section in knowledge_points:
                content += f"{section['heading']}\n"
                content += "=" * len(section['heading']) + "\n\n"
                
                for i, point in enumerate(section['points']):
                    content += f"{i+1}. {point['text']}\n\n"
    else:
        # 关键词模式或句子模式
        if output_format == "Markdown":
            content = "# PDF文档知识点提炼\n\n"
            
            for i, point in enumerate(knowledge_points):
                if point['type'] == "keyword":
                    content += f"- **{point['text']}**\n"
                else:
                    content += f"## 知识点 {i+1}\n\n{point['text']}\n\n"
        else:  # 纯文本
            content = "PDF文档知识点提炼\n\n"
            
            for i, point in enumerate(knowledge_points):
                if point['type'] == "keyword":
                    content += f"* {point['text']}\n"
                else:
                    content += f"{i+1}. {point['text']}\n\n"
    
    return content

# 主界面
st.markdown('<h1 class="main-header">PDF知识点提炼工具</h1>', unsafe_allow_html=True)
st.markdown('上传PDF文件，自动提取重要知识点，帮助您快速掌握文档内容。')

# 侧边栏参数设置
with st.sidebar:
    st.header("参数设置")
    
    importance = st.slider(
        "重要性阈值", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.1,
        help="调整该值控制提取知识点的数量。值越大，提取的知识点越少但更重要。"
    )
    
    mode = st.selectbox(
        "提取模式",
        ["自动模式", "关键词模式", "句子模式", "章节模式"],
        help="自动模式：分析文档结构选择最佳提取方式\n关键词模式：提取重要术语和关键词\n句子模式：提取重要句子\n章节模式：按章节组织提取"
    )
    
    output_format = st.selectbox(
        "输出格式",
        ["Markdown", "纯文本"],
        help="Markdown：格式化文本，支持层次结构\n纯文本：简单文本格式"
    )
    
    # 高级选项
    with st.expander("高级选项"):
        use_ocr = st.checkbox("启用OCR（对扫描版PDF）", value=True, 
                            help="对于扫描版PDF或无法直接提取文本的PDF使用光学字符识别")
        
        show_debug = st.checkbox("显示调试信息", value=False,
                               help="显示原始提取文本和处理过程")
    
    st.divider()
    
    st.write("**关于本工具**")
    st.write("本工具帮助您从PDF文档中提取重要知识点，支持文本PDF和扫描版PDF。")
    st.write("提取的知识点按重要性排序，可帮助您快速掌握文档核心内容。")

# 上传PDF文件
uploaded_file = st.file_uploader("选择PDF文件", type="pdf")

if uploaded_file is not None:
    # 显示文件信息
    file_details = {
        "文件名": uploaded_file.name,
        "文件大小": f"{uploaded_file.size / 1024:.1f} KB"
    }
    st.write(file_details)
    
    # 处理按钮
    if st.button("开始提取知识点"):
        with st.spinner("正在处理中，请稍候..."):
            try:
                # 提取文本
                if show_debug:
                    text, results, errors = extract_text_from_pdf(uploaded_file, use_ocr=use_ocr, debug=True)
                else:
                    text = extract_text_from_pdf(uploaded_file, use_ocr=use_ocr)
                
                # 调试信息
                if show_debug:
                    st.subheader("调试信息")
                    st.markdown("#### 提取方法")
                    for method, result in results.items():
                        with st.expander(f"{method} 提取结果"):
                            st.markdown(f'<div class="debug-info">{result[:2000]}{"..." if len(result) > 2000 else ""}</div>', unsafe_allow_html=True)
                    
                    if errors:
                        st.markdown("#### 错误信息")
                        for error in errors:
                            st.markdown(f'<div class="error-message">{error}</div>', unsafe_allow_html=True)
                    
                    st.markdown("#### 原始提取文本")
                    st.markdown(f'<div class="debug-info">{text[:3000]}{"..." if len(text) > 3000 else ""}</div>', unsafe_allow_html=True)
                
                # 如果文本提取成功
                if text and not text.startswith("无法提取文本"):
                    # 提取知识点
                    knowledge_points = extract_knowledge_points(text, mode, importance)
                    
                    if not knowledge_points or (mode != "章节模式" and len(knowledge_points) == 0) or \
                       (mode
