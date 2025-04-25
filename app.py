import streamlit as st
import io
import os
import re
import base64
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

# 页面配置
st.set_page_config(
    page_title="PDF多功能工具",
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
    .feature-option {
        padding: 10px;
        border-radius: 5px;
        background-color: #f5f5f5;
        margin: 5px 0;
        cursor: pointer;
    }
    .feature-option:hover {
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file, debug=False):
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
        
        # 如果提取的文本是URL或明显的乱码模式，给出警告
        if re.search(r'(https?:\/\/[^\s]+){3,}', text):
            errors.append("警告: 提取的文本可能包含大量URL或乱码，这可能是PDF格式问题")
    
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
        # 简单中文分词（字符级）
        words = list(sentence)
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
            # 简单的中文关键词提取（按字符频率）
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

def show_pdf_extractor():
    """显示PDF知识点提炼界面"""
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
            show_debug = st.checkbox("显示调试信息", value=False,
                                   help="显示原始提取文本和处理过程")
        
        st.divider()
        
        st.write("**关于本工具**")
        st.write("本工具帮助您从PDF文档中提取重要知识点，适用于文本型PDF。")
        st.info("注意：扫描版PDF可能无法正常提取文本。")
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
                        text, results, errors = extract_text_from_pdf(uploaded_file, debug=True)
                    else:
                        text = extract_text_from_pdf(uploaded_file)
                    
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
                           (mode == "章节模式" and len(knowledge_points) == 0):
                            st.warning("未能提取到足够的知识点，正在尝试不同的提取模式...")
                            # 尝试其他模式
                            if mode != "句子模式":
                                knowledge_points = extract_knowledge_points(text, "句子模式", max(0.3, importance - 0.2))
                            
                            if not knowledge_points or len(knowledge_points) == 0:
                                knowledge_points = extract_knowledge_points(text, "关键词模式", 0.3)
                                
                            if not knowledge_points or len(knowledge_points) == 0:
                                st.error("无法从文档中提取有效知识点。可能是文档格式问题或文本内容较少。")
                                st.stop()
                        
                        # 格式化输出
                        result_content = format_knowledge_points(knowledge_points, mode, output_format)
                        
                        # 创建结果区域
                        st.markdown('<h2 class="sub-header">提取结果</h2>', unsafe_allow_html=True)
                        
                        # 显示知识点
                        if mode == "章节模式":
                            for section in knowledge_points:
                                with st.expander(f"{section['heading']}"):
                                    for point in section['points']:
                                        st.markdown(f'<div class="knowledge-point">{point["text"]}</div>', unsafe_allow_html=True)
                        else:
                            # 关键词模式或句子模式
                            for point in knowledge_points[:20]:  # 最多显示20个知识点
                                if point['type'] == "keyword":
                                    st.markdown(f'<span class="highlight">{point["text"]}</span> ', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="knowledge-point">{point["text"]}</div>', unsafe_allow_html=True)
                            
                            if len(knowledge_points) > 20:
                                st.info(f"共提取了 {len(knowledge_points)} 个知识点，下载文件可查看全部内容。")
                        
                        # 下载链接
                        st.markdown("### 下载结果")
                        filename = f"{uploaded_file.name.split('.')[0]}_知识点.{'md' if output_format == 'Markdown' else 'txt'}"
                        download_link = create_downloadable_link(result_content, filename, "点击下载知识点提取结果")
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # 统计信息
                        if mode == "章节模式":
                            total_points = sum(len(s['points']) for s in knowledge_points)
                            st.sidebar.success(f"成功提取了 {len(knowledge_points)} 个章节和 {total_points} 个知识点")
                        else:
                            st.sidebar.success(f"成功提取了 {len(knowledge_points)} 个知识点")
                    
                    else:
                        # 文本提取失败
                        st.error("无法从PDF中提取有效文本。这可能是一个扫描版PDF或受保护的文档。")
                        st.info("目前版本不支持扫描版PDF的处理。请尝试使用文本型PDF文件。")
                        
                        # 显示提取的部分文本（如果有）
                        if text and show_debug:
                            st.markdown("#### 提取的部分文本")
                            st.markdown(f'<div class="debug-info">{text[:1000]}{"..." if len(text) > 1000 else ""}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"处理过程中出错: {str(e)}")
                    st.info("提示：如果是PDF格式问题，请尝试使用其他PDF文件。")
                    import traceback
                    if show_debug:
                        st.markdown("#### 错误详情")
                        st.markdown(f'<div class="error-message">{traceback.format_exc()}</div>', unsafe_allow_html=True)
    else:
        # 未上传文件时显示使用说明
        st.info("请上传PDF文件以开始提取知识点")
        
        with st.expander("使用指南"):
            st.markdown("""
            ### 基本使用流程
            
           ### 基本使用流程
            
            1. 在左侧上传一个PDF文件（支持中英文文档）
            2. 根据需要调整参数：
               - **重要性阈值**：控制提取的知识点数量和质量
               - **提取模式**：选择适合您文档的提取方式
               - **输出格式**：选择结果的格式化方式
            3. 在高级选项中，可以启用调试信息查看详细处理过程
            4. 点击"开始提取知识点"按钮
            5. 查看提取结果并下载
            
            ### 提取模式说明
            
            - **自动模式**：系统分析文档结构，选择最合适的提取方式
            - **关键词模式**：提取文档中的重要术语和关键词
            - **句子模式**：提取包含重要信息的完整句子
            - **章节模式**：按文档章节结构组织提取的知识点
            
            ### 重要性阈值
            
            - **0.1-0.3**：提取更多知识点，包括次要内容
            - **0.4-0.6**：平衡数量和质量
            - **0.7-0.9**：仅提取最重要的知识点
            
            ### 支持的PDF类型
            
            - 文本型PDF（如从Word导出的PDF）
            - 包含可选择文本的PDF
            - 注意：当前版本不支持扫描版PDF
            """)

        with st.expander("使用提示"):
            st.markdown("""
            ### 适合处理的文档
            
            ✅ 学术论文和研究报告  
            ✅ 技术文档和使用手册  
            ✅ 教材和学习资料  
            ✅ 企业报告和政策文件  
            ✅ 电子书和文章（文本型）  
            
            ### 不适合处理的文档
            
            ❌ 扫描版PDF（无文本层）  
            ❌ 主要由图表组成的文档  
            ❌ 密码保护或加密PDF  
            ❌ 格式非常复杂的PDF  
            
            ### 提高提取质量的技巧
            
            1. 确保PDF文件清晰，文本可选择
            2. 对于内容丰富的文档，适当降低重要性阈值
            3. 根据文档结构选择合适的提取模式
            4. 使用"自动模式"让系统自行判断最佳提取方式
            """)

def show_ppt_generator():
    """显示PPT生成器界面"""
    st.markdown('<h1 class="main-header">PDF生成PPT工具</h1>', unsafe_allow_html=True)
    st.markdown('上传PDF文件，自动转换为PPT演示文稿。')
    
    st.info("该功能将使用PDF提取的知识点自动创建PowerPoint演示文稿。")
    
    # 侧边栏参数
    with st.sidebar:
        st.header("PPT设置")
        
        ppt_theme = st.selectbox(
            "PPT主题",
            ["简约蓝", "商务灰", "学术绿", "鲜明红", "暗黑模式"],
            help="选择PPT的视觉主题"
        )
        
        slide_density = st.slider(
            "幻灯片内容密度", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="1=每张幻灯片内容较少，5=每张幻灯片内容较多"
        )
        
        include_toc = st.checkbox("包含目录页", value=True)
        include_cover = st.checkbox("包含封面", value=True)
        
    # 上传PDF文件
    uploaded_file = st.file_uploader("选择PDF文件", type="pdf")
    
    if uploaded_file is not None:
        file_details = {
            "文件名": uploaded_file.name,
            "文件大小": f"{uploaded_file.size / 1024:.1f} KB"
        }
        st.write(file_details)
        
        # 封面设置（可选）
        if include_cover:
            with st.expander("封面设置"):
                title = st.text_input("演示标题", value=uploaded_file.name.split('.')[0])
                subtitle = st.text_input("演示副标题", value="自动生成的演示文稿")
                author = st.text_input("作者", value="")
                date = st.date_input("日期")
        
        # 处理按钮
        if st.button("开始生成PPT"):
            with st.spinner("正在处理中，请稍候..."):
                try:
                    # 首先提取文本和知识点
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text and not text.startswith("无法提取文本"):
                        # 提取知识点（使用章节模式以获得更好的结构）
                        knowledge_points = extract_knowledge_points(text, "章节模式", 0.4)
                        
                        # 如果没有找到章节，尝试句子模式
                        if not knowledge_points or len(knowledge_points) == 0:
                            knowledge_points = extract_knowledge_points(text, "句子模式", 0.4)
                        
                        # 显示成功消息
                        st.success("已成功分析文档内容！")
                        
                        # 在实际应用中，这里会调用PPT生成模块
                        # 由于我们尚未集成实际的PPT生成功能，显示一个模拟界面
                        st.markdown("### 生成的PPT预览")
                        
                        # 模拟PPT预览
                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown("""
                            <div style="border:1px solid #ddd; padding:10px; text-align:center;">
                                <h4>封面</h4>
                                <p>演示标题</p>
                                <p>副标题</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with cols[1]:
                            st.markdown("""
                            <div style="border:1px solid #ddd; padding:10px; text-align:center;">
                                <h4>目录</h4>
                                <p>主要章节1</p>
                                <p>主要章节2</p>
                                <p>...</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with cols[2]:
                            st.markdown("""
                            <div style="border:1px solid #ddd; padding:10px; text-align:center;">
                                <h4>内容页</h4>
                                <p>主要知识点</p>
                                <p>支持要点</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 下载按钮（模拟）
                        st.info("提示：PPT生成功能仍在开发中，目前仅提供预览。")
                        if st.button("模拟下载PPT"):
                            st.success("在真实应用中，这里会提供一个PPT文件下载链接。")
                    else:
                        st.error("无法从PDF中提取有效文本。请尝试使用文本型PDF文件。")
                    
                except Exception as e:
                    st.error(f"处理过程中出错: {str(e)}")
                    st.info("提示：如果是PDF格式问题，请尝试使用其他PDF文件。")
    else:
        st.info("请上传PDF文件以生成PPT")
        
        with st.expander("功能说明"):
            st.markdown("""
            ### PDF转PPT功能
            
            本功能会自动分析您的PDF文档内容，提取重要知识点，并生成结构化的PPT演示文稿。
            
            **主要功能**：
            - 自动提取文档结构和内容
            - 根据文档结构创建幻灯片
            - 支持自定义PPT主题和风格
            - 可调整内容密度和展示方式
            
            **使用建议**：
            - 上传结构清晰的文本型PDF
            - 为获得最佳效果，选择包含清晰标题和小标题的文档
            - 调整内容密度以控制每张幻灯片的信息量
            """)

def show_animation_generator():
    """显示动画生成器界面"""
    st.markdown('<h1 class="main-header">知识点动画生成工具</h1>', unsafe_allow_html=True)
    st.markdown('上传PDF文件或输入文本，生成知识点讲解动画。')
    
    st.info("该功能可将PDF文档或文本内容转换为生动的动画讲解视频。")
    
    # 侧边栏参数
    with st.sidebar:
        st.header("动画设置")
        
        animation_style = st.selectbox(
            "动画风格",
            ["简约教学", "生动活泼", "专业商务", "科技感"],
            help="选择动画的视觉风格"
        )
        
        voice_type = st.selectbox(
            "配音风格",
            ["成熟男声", "亲和女声", "活力青年", "无配音"],
            help="选择讲解音频的配音类型"
        )
        
        animation_length = st.slider(
            "动画时长目标(分钟)", 
            min_value=1,
            max_value=10, 
            value=3,
            help="设置生成的动画大致时长"
        )
        
        include_background_music = st.checkbox("添加背景音乐", value=True)
    
    # 内容输入选项
    input_method = st.radio("选择输入方式", ["上传PDF", "直接输入文本"])
    
    if input_method == "上传PDF":
        uploaded_file = st.file_uploader("选择PDF文件", type="pdf")
        
        if uploaded_file is not None:
            file_details = {
                "文件名": uploaded_file.name,
                "文件大小": f"{uploaded_file.size / 1024:.1f} KB"
            }
            st.write(file_details)
            
            # 处理按钮
            if st.button("开始生成动画"):
                with st.spinner("正在处理中，请稍候..."):
                    try:
                        # 首先提取文本和知识点
                        text = extract_text_from_pdf(uploaded_file)
                        
                        if text and not text.startswith("无法提取文本"):
                            # 提取知识点
                            knowledge_points = extract_knowledge_points(text, "句子模式", 0.4)
                            
                            # 显示成功消息
                            st.success("已成功分析文档内容！")
                            
                            # 在实际应用中，这里会调用动画生成模块
                            # 由于我们尚未集成实际的动画生成功能，显示一个模拟界面
                            st.markdown("### 动画生成预览")
                            
                            # 模拟动画预览
                            st.markdown("""
                            <div style="background:#f0f0f0; padding:20px; border-radius:5px; text-align:center;">
                                <h4>动画预览区域</h4>
                                <p style="color:#555;">实际应用中，这里会显示动画预览或生成进度</p>
                                <div style="background:#ddd; height:240px; display:flex; align-items:center; justify-content:center;">
                                    <p>动画预览画面</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 下载按钮（模拟）
                            st.info("提示：动画生成功能仍在开发中，目前仅提供界面预览。")
                            if st.button("模拟下载动画"):
                                st.success("在真实应用中，这里会提供一个视频文件下载链接。")
                        else:
                            st.error("无法从PDF中提取有效文本。请尝试使用文本型PDF文件。")
                        
                    except Exception as e:
                        st.error(f"处理过程中出错: {str(e)}")
                        st.info("提示：如果是PDF格式问题，请尝试使用其他PDF文件。")
        else:
            st.info("请上传PDF文件以生成动画")
    
    else:  # 直接输入文本
        input_text = st.text_area("输入要转换为动画的文本内容", height=200)
        
        if st.button("开始生成动画") and input_text:
            with st.spinner("正在处理中，请稍候..."):
                try:
                    # 使用文本进行知识点提取
                    if len(input_text.strip()) > 10:
                        # 提取知识点
                        knowledge_points = extract_knowledge_points(input_text, "句子模式", 0.4)
                        
                        # 显示成功消息
                        st.success("已成功分析文本内容！")
                        
                        # 显示模拟界面（与PDF部分相同）
                        st.markdown("### 动画生成预览")
                        
                        # 模拟动画预览
                        st.markdown("""
                        <div style="background:#f0f0f0; padding:20px; border-radius:5px; text-align:center;">
                            <h4>动画预览区域</h4>
                            <p style="color:#555;">实际应用中，这里会显示动画预览或生成进度</p>
                            <div style="background:#ddd; height:240px; display:flex; align-items:center; justify-content:center;">
                                <p>动画预览画面</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 下载按钮（模拟）
                        st.info("提示：动画生成功能仍在开发中，目前仅提供界面预览。")
                        if st.button("模拟下载动画"):
                            st.success("在真实应用中，这里会提供一个视频文件下载链接。")
                    else:
                        st.error("输入文本太短，无法生成有意义的动画。")
                
                except Exception as e:
                    st.error(f"处理过程中出错: {str(e)}")
        
    with st.expander("功能说明"):
        st.markdown("""
        ### 知识点动画生成功能
        
        本功能可以将PDF文档或文本转换为生动的知识点讲解动画。
        
        **主要功能**：
        - 自动提取文档中的关键知识点
        - 将知识点转换为动画形式的讲解内容
        - 支持自定义动画风格和配音
        - 可选添加背景音乐
        
        **使用建议**：
        - 上传内容清晰的文本型PDF或直接输入文本
        - 内容最好以知识点讲解类型为主
        - 为获得最佳效果，控制输入内容的长度和复杂度
        """)

# 主函数
def main():
    # 在侧边栏添加功能选择
    with st.sidebar:
        st.title("PDF多功能工具")
        app_mode = st.radio(
            "选择功能",
            ["PDF知识点提炼", "生成PPT", "生成动画"],
            help="选择您想要使用的功能"
        )
        st.divider()
    
    # 根据选择加载不同功能
    if app_mode == "PDF知识点提炼":
        show_pdf_extractor()
    elif app_mode == "生成PPT":
        show_ppt_generator()
    elif app_mode == "生成动画":
        show_animation_generator()

    # 添加页脚
    st.markdown("""
    ---
    <p style="text-align: center; color: gray; font-size: 0.8em;">
    PDF多功能工具 | 版本 1.0 | © 2025
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
