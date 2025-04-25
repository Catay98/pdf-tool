"""
内容理解与知识提取模块
------------------
用于分析和理解PDF处理后提取的文本内容，识别关键概念、
知识点，并构建知识图谱。这是连接PDF处理和内容生成的桥梁。
"""

import os
import json
import time
import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 尝试导入LLM相关库
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentUnderstanding:
    """内容理解与知识提取主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化内容理解模块
        
        Args:
            config: 配置参数，包括API密钥、模型选择等
        """
        self.config = config or {}
        
        # 初始化NLP工具
        self.nlp = spacy.load("zh_core_web_sm")
        
        # 初始化LLM客户端
        self.llm_client = self._init_llm_client()
        
        # 学科特定关键词
        self.subject_keywords = self._load_subject_keywords()
    
    def _init_llm_client(self):
        """初始化大语言模型客户端"""
        llm_provider = self.config.get("llm_provider", "openai")
        
        if llm_provider == "openai" and OPENAI_AVAILABLE:
            api_key = self.config.get("openai_api_key")
            if api_key:
                return OpenAI(api_key=api_key)
            else:
                logger.warning("未提供OpenAI API密钥，将使用本地模式")
                return None
        else:
            logger.warning(f"不支持的LLM提供商: {llm_provider}，将使用本地模式")
            return None
    
    def _load_subject_keywords(self):
        """加载学科特定关键词"""
        # 这里可以从文件加载，或者使用预定义的词典
        # 示例: 科学教育相关词汇
        return {
            "physics": ["力", "质量", "加速度", "能量", "动量", "功率", "电场", "磁场", "波动", "热力学",
                        "牛顿", "爱因斯坦", "光速", "相对论", "量子", "原子", "电子", "质子", "中子"],
            "chemistry": ["元素", "分子", "化合物", "反应", "催化剂", "酸", "碱", "氧化", "还原", "浓度",
                        "溶液", "电解质", "离子", "价键", "周期表", "原子量", "摩尔", "化学平衡"],
            "biology": ["细胞", "基因", "蛋白质", "酶", "DNA", "RNA", "进化", "生态", "系统", "遗传",
                     "变异", "光合作用", "呼吸作用", "组织", "器官", "系统", "微生物"],
            "math": ["函数", "极限", "导数", "积分", "微分", "方程", "不等式", "概率", "统计", "矩阵", 
                   "向量", "几何", "代数", "集合", "数列", "级数", "三角", "对数"]
        }
    
    def analyze_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析处理后的PDF内容
        
        Args:
            content_data: PDFProcessor处理后的结构化内容
            
        Returns:
            分析结果，包括关键概念、知识图谱等
        """
        logger.info("开始分析内容...")
        
        # 合并所有章节文本用于整体分析
        all_text = self._extract_all_text(content_data)
        
        # 初始化分析结果
        analysis_result = {
            "subject": None,
            "main_topics": [],
            "chapters": [],
            "knowledge_graph": {
                "nodes": [],
                "edges": []
            },
            "difficulty_level": None,
            "target_audience": None,
            "important_concepts": []
        }
        
        # 步骤1: 识别学科领域
        analysis_result["subject"] = self._identify_subject(all_text)
        logger.info(f"识别学科: {analysis_result['subject']}")
        
        # 步骤2: 分析每个章节
        for chapter in content_data.get("chapters", []):
            chapter_analysis = self._analyze_chapter(chapter, analysis_result["subject"])
            analysis_result["chapters"].append(chapter_analysis)
            
            # 收集主题
            analysis_result["main_topics"].extend(chapter_analysis.get("main_topics", []))
        
        # 去重主题
        analysis_result["main_topics"] = list(set(analysis_result["main_topics"]))
        
        # 步骤3: 构建知识图谱
        analysis_result["knowledge_graph"] = self._build_knowledge_graph(analysis_result["chapters"])
        
        # 步骤4: 使用LLM进行高级分析
        if self.llm_client:
            llm_analysis = self._perform_llm_analysis(content_data)
            # 更新分析结果
            analysis_result.update(llm_analysis)
        
        # 步骤5: 识别重要概念
        analysis_result["important_concepts"] = self._identify_important_concepts(
            analysis_result["chapters"], 
            analysis_result.get("subject")
        )
        
        logger.info("内容分析完成")
        return analysis_result
    
    def _extract_all_text(self, content_data: Dict[str, Any]) -> str:
        """从内容数据中提取所有文本"""
        all_texts = []
        
        for chapter in content_data.get("chapters", []):
            # 添加章节标题
            all_texts.append(chapter.get("title", ""))
            
            # 添加章节内容
            for item in chapter.get("content", []):
                if "text" in item:
                    all_texts.append(item["text"])
                    
            # 添加子章节内容
            for subchapter in chapter.get("subchapters", []):
                all_texts.append(subchapter.get("title", ""))
                for item in subchapter.get("content", []):
                    if "text" in item:
                        all_texts.append(item["text"])
        
        return " ".join(all_texts)
    
    def _identify_subject(self, text: str) -> str:
        """识别文本的学科领域"""
        subject_scores = {}
        
        # 对每个学科计算关键词匹配度
        for subject, keywords in self.subject_keywords.items():
            score = 0
            for keyword in keywords:
                matches = re.findall(rf'\b{keyword}\b', text, re.IGNORECASE)
                score += len(matches)
            
            subject_scores[subject] = score
        
        # 选择得分最高的学科
        if subject_scores:
            max_subject = max(subject_scores.items(), key=lambda x: x[1])
            if max_subject[1] > 0:
                return max_subject[0]
        
        # 如果没有明确匹配，使用LLM识别（如果可用）
        if self.llm_client:
            return self._llm_identify_subject(text[:5000])  # 限制文本长度
            
        return "general"  # 默认为通用学科
    
    def _llm_identify_subject(self, text: str) -> str:
        """使用LLM识别文本的学科领域"""
        try:
            prompt = f"""请分析以下教材文本，确定其属于哪个学科领域。
            请从以下选项中选择最匹配的一个：物理(physics)、化学(chemistry)、生物(biology)、数学(math)、计算机科学(computer_science)、
            语文(chinese)、历史(history)、地理(geography)、政治(politics)、通用(general)
            
            文本摘录：
            {text[:3000]}
            
            学科："""
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专门识别教材学科领域的助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            
            # 提取学科
            response_text = response.choices[0].message.content.strip().lower()
            
            # 匹配可能的学科词
            subjects = {
                "physics": "physics", "物理": "physics",
                "chemistry": "chemistry", "化学": "chemistry",
                "biology": "biology", "生物": "biology",
                "math": "math", "数学": "math",
                "computer science": "computer_science", "计算机": "computer_science",
                "chinese": "chinese", "语文": "chinese",
                "history": "history", "历史": "history",
                "geography": "geography", "地理": "geography",
                "politics": "politics", "政治": "politics"
            }
            
            for key, value in subjects.items():
                if key in response_text:
                    return value
                    
            return "general"
            
        except Exception as e:
            logger.error(f"LLM识别学科失败: {e}")
            return "general"
    
    def _analyze_chapter(self, chapter: Dict[str, Any], subject: str) -> Dict[str, Any]:
        """
        分析单个章节的内容
        
        Args:
            chapter: 章节数据
            subject: 已识别的学科
            
        Returns:
            章节分析结果
        """
        chapter_title = chapter.get("title", "")
        chapter_content = []
        
        # 提取章节所有文本内容
        for item in chapter.get("content", []):
            if "text" in item:
                chapter_content.append(item["text"])
        
        chapter_text = " ".join(chapter_content)
        
        # 使用spaCy分析文本
        doc = self.nlp(chapter_text[:100000])  # 限制长度避免内存问题
        
        # 分析结果
        chapter_analysis = {
            "title": chapter_title,
            "key_concepts": [],
            "main_topics": [],
            "summary": "",
            "difficulty_level": None,
            "learning_objectives": []
        }
        
        # 提取关键概念 (使用实体识别和名词短语)
        key_concepts = []
        
        # 实体识别
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE", "LAW", "WORK_OF_ART", "EVENT", "FAC"]:
                key_concepts.append(ent.text)
        
        # 名词短语识别
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 1:  # 过滤单字名词
                noun_phrases.append(chunk.text)
        
        # 合并并去重
        key_concepts.extend(noun_phrases)
        key_concepts = list(set(key_concepts))
        
        # 使用TF-IDF找出该章节特有的关键词
        if len(chapter_content) > 0:
            tfidf_keywords = self._extract_tfidf_keywords(chapter_content, 10)
            key_concepts.extend(tfidf_keywords)
        
        # 过滤关键概念
        filtered_concepts = []
        for concept in key_concepts:
            # 过滤太短的概念
            if len(concept) < 2:
                continue
                
            # 过滤停用词
            if concept.strip() in self.nlp.Defaults.stop_words:
                continue
                
            # 添加到过滤后列表
            filtered_concepts.append(concept)
            
        # 取前20个关键概念
        chapter_analysis["key_concepts"] = filtered_concepts[:20]
        
        # 提取主题
        chapter_analysis["main_topics"] = self._extract_main_topics(chapter_text, subject)
        
        # 生成摘要
        chapter_analysis["summary"] = self._generate_summary(chapter_text)
        
        # 估计难度级别
        chapter_analysis["difficulty_level"] = self._estimate_difficulty(chapter_text, subject)
        
        # 如果有子章节，递归分析
        if "subchapters" in chapter:
            subchapter_analyses = []
            for subchapter in chapter["subchapters"]:
                sub_analysis = self._analyze_chapter(subchapter, subject)
                subchapter_analyses.append(sub_analysis)
            
            chapter_analysis["subchapters"] = subchapter_analyses
        
        # 如果可用，使用LLM辅助分析
        if self.llm_client:
            llm_chapter_analysis = self._llm_analyze_chapter(chapter_title, chapter_text[:4000], subject)
            
            # 合并LLM分析结果
            if "learning_objectives" in llm_chapter_analysis:
                chapter_analysis["learning_objectives"] = llm_chapter_analysis["learning_objectives"]
                
            if "summary" in llm_chapter_analysis and llm_chapter_analysis["summary"]:
                chapter_analysis["summary"] = llm_chapter_analysis["summary"]
        
        return chapter_analysis
    
    def _extract_tfidf_keywords(self, texts: List[str], top_n: int = 10) -> List[str]:
        """使用TF-IDF算法提取文本中的关键词"""
        if not texts:
            return []
            
        try:
            # 创建TF-IDF向量器
            vectorizer = TfidfVectorizer(
                max_features=100,
                token_pattern=r'(?u)\b\w+\b',
                stop_words=list(self.nlp.Defaults.stop_words)
            )
            
            # 转换文本
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 获取词汇表
            feature_names = vectorizer.get_feature_names_out()
            
            # 计算每个词的平均TF-IDF分数
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # 获取前N个关键词
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"TF-IDF关键词提取失败: {e}")
            return []
    
    def _extract_main_topics(self, text: str, subject: str) -> List[str]:
        """从文本中提取主题"""
        # 基于学科关键词和文本分析提取主题
        
        # 使用学科特定关键词
        subject_kws = self.subject_keywords.get(subject, [])
        
        # 在文本中查找这些关键词
        found_keywords = []
        for kw in subject_kws:
            if re.search(rf'\b{kw}\b', text, re.IGNORECASE):
                found_keywords.append(kw)
        
        # 使用spaCy的命名实体识别查找其他可能的主题
        doc = self.nlp(text[:50000])  # 限制处理的文本量
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "EVENT", "WORK_OF_ART"]]
        
        # 合并并去重
        topics = found_keywords + entities
        unique_topics = list(set(topics))
        
        return unique_topics[:10]  # 返回前10个主题
    
    def _generate_summary(self, text: str) -> str:
        """生成文本摘要"""
        # 简单实现：使用前几句话作为摘要
        sentences = re.split(r'[。！？.!?]', text)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        
        if valid_sentences:
            summary = "。".join(valid_sentences[:3]) + "。"
            return summary
        
        return ""
    
    def _estimate_difficulty(self, text: str, subject: str) -> str:
        """估计内容的难度级别"""
        # 简单的难度估计算法
        
        # 1. 句子长度因素
        sentences = re.split(r'[。！？.!?]', text)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        
        if not valid_sentences:
            return "unknown"
            
        avg_sentence_length = sum(len(s) for s in valid_sentences) / len(valid_sentences)
        
        # 2. 专业术语因素
        subject_terms = set(self.subject_keywords.get(subject, []))
        term_count = sum(1 for term in subject_terms if term in text)
        
        # 3. 复杂句型因素 (包含特定标点的句子比例)
        complex_punct = ["：", "；", "，", "、", "「", "」", "（", "）", ":", ";", ",", "(", ")"]
        complex_sentences = sum(1 for s in valid_sentences if any(p in s for p in complex_punct))
        complex_ratio = complex_sentences / len(valid_sentences) if valid_sentences else 0
        
        # 难度评分计算
        difficulty_score = (
            (avg_sentence_length / 15) * 0.4 +  # 句子长度因素
            (term_count / (len(subject_terms) + 1)) * 0.3 +  # 专业术语因素
            complex_ratio * 0.3  # 复杂句型因素
        )
        
        # 难度映射
        if difficulty_score < 0.3:
            return "beginner"
        elif difficulty_score < 0.6:
            return "intermediate"
        else:
            return "advanced"
    
    def _llm_analyze_chapter(self, title: str, text: str, subject: str) -> Dict[str, Any]:
        """使用LLM分析章节内容"""
        try:
            prompt = f"""请分析以下教材章节内容，并提供:
            1. 学习目标 (Learning Objectives): 列出3-5个该章节的学习目标
            2. 章节摘要 (Summary): 100-150字的章节内容摘要
            
            章节标题: {title}
            学科: {subject}
            
            章节内容:
            {text[:4000]}
            
            请用JSON格式回答，包含learning_objectives(数组)和summary(字符串)字段。
            """
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专门分析教材内容的助手。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # 解析响应
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"LLM章节分析失败: {e}")
            return {
                "learning_objectives": [],
                "summary": ""
            }
    
    def _build_knowledge_graph(self, chapter_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        构建知识图谱
        
        Args:
            chapter_analyses: 章节分析结果列表
            
        Returns:
            知识图谱数据
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 收集所有概念
        all_concepts = []
        chapter_concepts = {}
        
        for chapter in chapter_analyses:
            chapter_title = chapter.get("title", "")
            concepts = chapter.get("key_concepts", [])
            
            all_concepts.extend(concepts)
            chapter_concepts[chapter_title] = concepts
            
            # 将章节添加为节点
            G.add_node(chapter_title, type="chapter")
            
            # 将概念添加为节点，并连接到章节
            for concept in concepts:
                G.add_node(concept, type="concept")
                G.add_edge(chapter_title, concept, type="contains")
        
        # 计算概念间的相似度并添加边
        self._add_concept_relationships(G, all_concepts)
        
        # 添加从前一章到后一章的顺序关系
        for i in range(len(chapter_analyses) - 1):
            current_chapter = chapter_analyses[i].get("title", "")
            next_chapter = chapter_analyses[i + 1].get("title", "")
            G.add_edge(current_chapter, next_chapter, type="next")
        
        # 转换为可序列化格式
        nodes = []
        for node, data in G.nodes(data=True):
            nodes.append({
                "id": node,
                "type": data.get("type", "unknown")
            })
        
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "type": data.get("type", "related")
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _add_concept_relationships(self, G: nx.DiGraph, concepts: List[str]) -> None:
        """
        基于相似度添加概念间的关系
        
        Args:
            G: 知识图谱
            concepts: 概念列表
        """
        if len(concepts) < 2:
            return
            
        # 计算概念之间的相似度
        try:
            # 使用spaCy文档向量计算相似度
            concept_docs = [self.nlp(concept) for concept in concepts]
            
            for i in range(len(concepts)):
                for j in range(i+1, len(concepts)):
                    # 只有当两个概念都有向量表示时才计算相似度
                    if concept_docs[i].vector_norm and concept_docs[j].vector_norm:
                        similarity = concept_docs[i].similarity(concept_docs[j])
                        
                        # 只添加相似度高的关系
                        if similarity > 0.6:  # 相似度阈值
                            G.add_edge(concepts[i], concepts[j], type="related", weight=similarity)
        except Exception as e:
            logger.warning(f"计算概念相似度时出错: {e}")
    
    def _perform_llm_analysis(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM进行高级内容分析
        
        Args:
            content_data: PDF处理和基础分析后的内容数据
            
        Returns:
            LLM高级分析结果
        """
        if not self.llm_client:
            return {}
        
        try:
            # 构建文本摘要用于LLM分析
            summary_text = self._build_content_summary(content_data)
            
            prompt = f"""作为教材分析专家，请分析以下教材内容摘要，并提供:
            1. 难度级别 (difficulty_level): 初级(beginner)/中级(intermediate)/高级(advanced)
            2. 目标受众 (target_audience): 该教材适合哪类学生
            3. 教学建议 (teaching_suggestions): 3-5条教学建议
            
            教材内容摘要:
            {summary_text}
            
            请用JSON格式回答，包含difficulty_level, target_audience和teaching_suggestions字段。
            """
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的教材分析专家，擅长分析教科书内容并提供教学建议。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # 解析响应
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"LLM高级分析失败: {e}")
            return {}
    
    def _build_content_summary(self, content_data: Dict[str, Any]) -> str:
        """为LLM分析构建内容摘要"""
        summary_parts = []
        
        # 添加章节标题
        chapter_titles = [f"章节{i+1}: {chapter.get('title', '未命名章节')}" 
                         for i, chapter in enumerate(content_data.get("chapters", []))]
        
        if chapter_titles:
            summary_parts.append("教材章节结构:\n" + "\n".join(chapter_titles))
        
        # 为每个章节添加内容摘要
        chapter_summaries = []
        for i, chapter in enumerate(content_data.get("chapters", [])[:3]):  # 只取前3章
            title = chapter.get("title", f"章节{i+1}")
            content_sample = ""
            
            # 提取一些内容示例
            content_items = chapter.get("content", [])
            if content_items:
                # 取前3段内容示例
                content_texts = [item.get("text", "") for item in content_items if "text" in item][:3]
                if content_texts:
                    content_sample = "\n".join(content_texts)
            
            if content_sample:
                chapter_summaries.append(f"章节: {title}\n内容示例:\n{content_sample[:500]}...")
        
        if chapter_summaries:
            summary_parts.append("内容示例:\n" + "\n\n".join(chapter_summaries))
        
        return "\n\n".join(summary_parts)
    
    def _identify_important_concepts(self, chapter_analyses: List[Dict[str, Any]], subject: str) -> List[Dict[str, Any]]:
        """
        识别整个内容中的重要概念
        
        Args:
            chapter_analyses: 章节分析结果
            subject: 学科
            
        Returns:
            重要概念列表，包含概念名称和重要性得分
        """
        # 收集所有概念
        all_concepts = []
        for chapter in chapter_analyses:
            all_concepts.extend(chapter.get("key_concepts", []))
        
        # 计算每个概念的出现频率
        concept_counter = Counter(all_concepts)
        
        # 找出频率最高的概念
        important_concepts = []
        for concept, count in concept_counter.most_common(20):
            important_concepts.append({
                "concept": concept,
                "importance": count,
                "chapters": [chapter.get("title") for chapter in chapter_analyses 
                            if concept in chapter.get("key_concepts", [])]
            })
        
        return important_concepts
    
    def visualize_knowledge_graph(self, knowledge_graph: Dict[str, Any], output_path: str = "knowledge_graph.png") -> None:
        """
        可视化知识图谱
        
        Args:
            knowledge_graph: 知识图谱数据
            output_path: 输出图像路径
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for node in knowledge_graph.get("nodes", []):
            node_id = node.get("id", "")
            node_type = node.get("type", "unknown")
            
            if node_id:
                G.add_node(node_id, type=node_type)
        
        # 添加边
        for edge in knowledge_graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("type", "related")
            
            if source and target and source in G and target in G:
                G.add_edge(source, target, type=edge_type)
        
        if not G.nodes():
            logger.warning("知识图谱为空，无法生成可视化")
            return
        
        # 设置节点颜色
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "")
            if node_type == "chapter":
                node_colors.append("lightblue")
            elif node_type == "concept":
                node_colors.append("lightgreen")
            else:
                node_colors.append("gray")
        
        # 设置边颜色
        edge_colors = []
        for u, v, data in G.edges(data=True):
            edge_type = data.get("type", "")
            if edge_type == "contains":
                edge_colors.append("blue")
            elif edge_type == "related":
                edge_colors.append("green")
            elif edge_type == "next":
                edge_colors.append("red")
            else:
                edge_colors.append("gray")
        
        # 创建可视化
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.6, 
                              arrowstyle='->', arrowsize=15)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
        
        # 添加图例
        node_types = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='章节'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='概念')
        ]
        
        edge_types = [
            plt.Line2D([0], [0], color='blue', lw=2, label='包含'),
            plt.Line2D([0], [0], color='green', lw=2, label='相关'),
            plt.Line2D([0], [0], color='red', lw=2, label='顺序')
        ]
        
        plt.legend(handles=node_types + edge_types, loc='upper right')
        
        plt.title("教材内容知识图谱")
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"知识图谱可视化已保存至: {output_path}")

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='分析教材内容并提取知识结构')
    parser.add_argument('input_file', help='输入的结构化内容JSON文件(由PDFProcessor生成)')
    parser.add_argument('--output', '-o', help='输出分析结果的JSON文件路径')
    parser.add_argument('--visualize', '-v', action='store_true', help='是否生成知识图谱可视化')
    parser.add_argument('--openai_api_key', help='OpenAI API密钥')
    
    args = parser.parse_args()
    
    # 加载输入数据
    with open(args.input_file, 'r', encoding='utf-8') as f:
        content_data = json.load(f)
    
    # 配置
    config = {}
    if args.openai_api_key:
        config['llm_provider'] = 'openai'
        config['openai_api_key'] = args.openai_api_key
    
    # 初始化内容理解模块
    content_understanding = ContentUnderstanding(config)
    
    # 分析内容
    analysis_result = content_understanding.analyze_content(content_data)
    
    # 设置默认输出路径
    output_path = args.output
    if not output_path:
        input_base = os.path.splitext(os.path.basename(args.input_file))[0]
        output_path = f"{input_base}_analysis.json"
    
    # 保存分析结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    print(f"内容分析结果已保存至: {output_path}")
    
    # 可视化知识图谱
    if args.visualize and analysis_result.get("knowledge_graph"):
        graph_path = os.path.splitext(output_path)[0] + "_graph.png"
        content_understanding.visualize_knowledge_graph(analysis_result["knowledge_graph"], graph_path)

if __name__ == "__main__":
    main()