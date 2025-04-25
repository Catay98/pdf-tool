import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
from collections import Counter
import base64

# 初始化NLTK资源（首次运行时下载）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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
</style>
""", unsafe_allow_html=True)

def preprocess_text(text):
    """预处理文本"""
    # 移除特殊字符和多余空格
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\?\!]', '', text)
    return text

def extract_sentences(text, min_length=5):
    """提取文本中的句子"""
    sentences = sent_tokenize(text)
    # 过滤太短的句子
    return [s for s in sentences if len(word_tokenize(s)) >= min_length]

def score_importance(sentence, keywords=None, stop_words=None):
    """评估句子的重要性"""
    if stop_words is None:
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
    
    words = word_tokenize(sentence.lower())
    # 过滤停用词
    filtered_words = [w for w in words if w not in stop_words and w.isalnum()]
    
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
    if any(w.isdigit() for w in words):
        score += 0.05
    
    # 特殊标记词语，通常表示重要内容
    importance_markers = ["important", "key", "significant", "essential", "crucial", "critical"]
    if any(marker in filtered_words for marker in importance_markers):
        score += 0.1
    
    return min(score, 1.0)  # 确保分数不超过1.0

def extract_keywords(text, top_n=20):
    """提取文本中的关键词"""
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
        
    words = word_tokenize(text.lower())
    filtered_words = [w for w in words if w not in stop_words and w.isalnum() and len(w) > 2]
    
    # 统计词频
    word_counts = Counter(filtered_words)
    
    # 返回最常见的词
    return [word for word, _ in word_counts.most_common(top_n)]

def get_chapter_headings(text):
    """尝试提取章节标题"""
    # 匹配常见的章节标题模式
    patterns = [
        r'(?:Chapter|CHAPTER)\s+\d+[:\.\s]+(.+?)(?:\n|\r\n?)',
        r'(?:Section|SECTION)\s+\d+(?:\.\d+)*[:\.\s]+(.+?)(?:\n|\r\n?)',
        r'^\d+(?:\.\d+)*[:\.\s]+(.+?)(?:\n|\r\n?)'
    ]
    
    headings = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        headings.extend(matches)
    
    return headings

def create_downloadable_link(content, filename, link_text):
    """创建可下载内容的链接"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def extract_knowledge_points(text, mode, importance_threshold):
    """根据不同模式提取知识点"""
    text = preprocess_text(text)
    
    if mode == "自动模式":
        # 检测文档结构，选择适合的模式
        headings = get_chapter_headings(text)
        if len(headings) > 3:
            mode = "章节模式"
        else:
            mode = "句子模式"
    
    # 提取关键词
    keywords = extract_keywords(text)
    
    if mode == "关键词模式":
        # 直接返回关键词作为知识点
        return [{"text": kw, "importance": 0.7, "type": "keyword"} for kw in keywords]
    
    elif mode == "句子模式":
        # 提取和评分句子
        sentences = extract_sentences(text)
        knowledge_points = []
        
        for sentence in sentences:
            importance = score_importance(sentence, keywords)
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
        sections = []
        
        # 简单分块
        chunks = re.split(r'(?:Chapter|CHAPTER|Section|SECTION)\s+\d+', text)
        
        # 处理每个章节
        for i, heading in enumerate(headings):
            section_text = chunks[i+1] if i+1 < len(chunks) else ""
            # 为每个章节提取关键句子
            sentences = extract_sentences(section_text)
            section_points = []
            
            for sentence in sentences:
                importance = score_importance(sentence, keywords)
                if importance >= importance_threshold:
                    section_points.append({
                        "text": sentence,
                        "importance": importance,
                        "type": "sentence"
                    })
            
            # 按重要性降序排序
            section_points = sorted(section_points, key=lambda x: x["importance"], reverse=True)
            
            sections.append({
                "heading": heading,
                "importance": 0.9,  # 章节标题通常很重要
                "type": "heading",
                "points": section_points[:5]  # 每章取前5个重要句子
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
                    content += f"- {point['text']}\n"
                
                content += "\n"
        else:  # 纯文本
            content = "PDF文档知识点提炼\n\n"
            
            for section in knowledge_points:
                content += f"{section['heading']}\n"
                content += "=" * len(section['heading']) + "\n\n"
                
                for i, point in enumerate(section['points']):
                    content += f"{i+1}. {point['text']}\n"
                
                content += "\n"
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
    
    st.divider()
    
    st.write("**关于本工具**")
    st.write("本工具帮助您从PDF文档中提取重要知识点，节省阅读时间。")
    st.write("由于是简化版本，提取效果可能不够精确，仅供参考。")

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
                # 读取PDF
                reader = PyPDF2.PdfReader(uploaded_file)
                
                # 提取文本
                text = ""
                total_pages = len(reader.pages)
                
                progress_bar = st.progress(0)
                for i, page in enumerate(reader.pages):
                    text += page.extract_text() or ""
                    progress_bar.progress((i + 1) / total_pages)
                
                # 提取知识点
                knowledge_points = extract_knowledge_points(text, mode, importance)
                
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
                    for point in knowledge_points:
                        if point['type'] == "keyword":
                            st.markdown(f'<span class="highlight">{point["text"]}</span> ', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="knowledge-point">{point["text"]}</div>', unsafe_allow_html=True)
                
                # 下载链接
                st.markdown("### 下载结果")
                filename = f"{uploaded_file.name.split('.')[0]}_知识点.{'md' if output_format == 'Markdown' else 'txt'}"
                download_link = create_downloadable_link(result_content, filename, "点击下载知识点提取结果")
                st.markdown(download_link, unsafe_allow_html=True)
                
                # 统计信息
                st.sidebar.success(f"成功提取了{len(knowledge_points) if mode != '章节模式' else sum(len(s['points']) for s in knowledge_points)}个知识点")
                
            except Exception as e:
                st.error(f"处理过程中出错: {str(e)}")
                st.info("提示：如果是PDF格式问题，请尝试使用其他PDF文件。")
else:
    # 未上传文件时显示使用说明
    st.info("请上传PDF文件以开始提取知识点")
    
    with st.expander("使用指南"):
        st.markdown("""
        ### 基本使用流程
        
        1. 在左侧上传一个PDF文件（支持中英文）
        2. 根据需要调整参数：
           - **重要性阈值**：控制提取的知识点数量和质量
           - **提取模式**：选择适合您文档的提取方式
           - **输出格式**：选择结果的格式化方式
        3. 点击"开始提取知识点"按钮
        4. 查看提取结果并下载
        
        ### 提取模式说明
        
        - **自动模式**：系统分析文档结构，选择最合适的提取方式
        - **关键词模式**：提取文档中的重要术语和关键词
        - **句子模式**：提取包含重要信息的完整句子
        - **章节模式**：按文档章节结构组织提取的知识点
        
        ### 重要性阈值
        
        - **0.1-0.3**：提取更多知识点，包括次要内容
        - **0.4-0.6**：平衡数量和质量
        - **0.7-0.9**：仅提取最重要的知识点
        """)

# 添加页脚
st.markdown("""
---
<p style="text-align: center; color: gray; font-size: 0.8em;">
PDF知识点提炼工具 | 版本 1.0 | © 2025
</p>
""", unsafe_allow_html=True)