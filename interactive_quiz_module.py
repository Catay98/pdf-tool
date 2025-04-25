"""
交互式测验与知识检验模块
---------------
用于根据教材内容自动生成测验题目，
提供即时反馈，并支持知识点掌握评估。
"""

import os
import json
import random
import re
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

# 尝试导入LLM相关库
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI API 库未安装，某些功能将受限")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuizGenerator:
    """测验题目生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化LLM客户端
        self.llm_client = self._init_llm_client()
        
        # 题型模板
        self.quiz_templates = self._load_quiz_templates()
        
        # 难度系数
        self.difficulty_weights = {
            'beginner': 0.6,
            'intermediate': 1.0,
            'advanced': 1.5
        }
    
    def _init_llm_client(self):
        """初始化大语言模型客户端"""
        llm_provider = self.config.get("llm_provider", "openai")
        
        if llm_provider == "openai" and OPENAI_AVAILABLE:
            api_key = self.config.get("openai_api_key")
            if api_key:
                return OpenAI(api_key=api_key)
            else:
                logger.warning("未提供OpenAI API密钥，将使用模板生成测验")
                return None
        else:
            logger.warning(f"不支持的LLM提供商: {llm_provider}，将使用模板生成测验")
            return None
    
    def _load_quiz_templates(self) -> Dict[str, List[str]]:
        """加载题型模板"""
        return {
            'multiple_choice': [
                "以下关于{concept}的描述，哪一项是正确的？",
                "{concept}的主要特点是什么？",
                "下列选项中，哪一个最准确地描述了{concept}？"
            ],
            'true_false': [
                "{statement}，这种说法是否正确？",
                "判断以下说法是否正确：{statement}",
                "下面的陈述是对是错：{statement}"
            ],
            'fill_blank': [
                "请填写空白：{concept}的定义是___。",
                "{statement_with_blank}",
                "在{context}中，我们使用___来表示{concept}。"
            ],
            'short_answer': [
                "简要解释{concept}的含义及其重要性。",
                "请简述{concept}与{related_concept}之间的关系。",
                "用一到两句话描述{concept}的应用场景。"
            ],
            'definition': [
                "请给出{concept}的准确定义。",
                "什么是{concept}？请给出明确定义。",
                "{concept}在{subject}中指的是什么？"
            ]
        }
    
    def generate_quizzes(self, content_analysis: Dict[str, Any], num_questions: int = 10) -> List[Dict[str, Any]]:
        """
        根据内容分析生成测验题目
        
        Args:
            content_analysis: 内容分析结果
            num_questions: 生成的题目数量
            
        Returns:
            测验题目列表
        """
        # 获取学科和难度
        subject = content_analysis.get('subject', 'general')
        global_difficulty = content_analysis.get('difficulty_level', 'intermediate')
        
        # 收集重要概念
        important_concepts = content_analysis.get('important_concepts', [])
        
        # 收集章节内容
        chapters = content_analysis.get('chapters', [])
        
        # 使用LLM生成题目
        if self.llm_client:
            return self._generate_llm_quizzes(content_analysis, num_questions)
        
        # 使用模板生成题目
        return self._generate_template_quizzes(
            important_concepts, 
            chapters, 
            subject, 
            global_difficulty, 
            num_questions
        )
def _generate_llm_quizzes(self, content_analysis: Dict[str, Any], num_questions: int) -> List[Dict[str, Any]]:
        """使用LLM生成测验题目"""
        try:
            # 提取重要内容
            subject = content_analysis.get('subject', 'general')
            important_concepts = content_analysis.get('important_concepts', [])
            
            # 为LLM构建提示词
            concept_list = ", ".join([concept.get('concept', '') for concept in important_concepts[:10]])
            
            prompt = f"""作为一位{subject}学科的资深教师，请根据以下教材重点内容创建{num_questions}道测验题，用于考察学生对知识点的掌握情况。

主要知识点：{concept_list}

请创建多种类型的题目，包括：
1. 多选题 (multiple_choice): 提供4个选项，其中只有1个正确答案
2. 判断题 (true_false): 学生需要判断陈述是对还是错
3. 填空题 (fill_blank): 句子中留有空白需要填写
4. 简答题 (short_answer): 需要学生简要回答的开放性问题

对于每个题目，请提供：
- 题目类型 (type)
- 题目文本 (question)
- 答案 (answer)
- 答案解析 (explanation)
- 相关概念 (concept)
- 难度级别 (difficulty): beginner, intermediate, 或 advanced

请以JSON格式输出，格式如下：
[
  {{
    "type": "multiple_choice",
    "question": "问题内容",
    "options": ["选项A", "选项B", "选项C", "选项D"],
    "answer": "正确选项的索引(0-3)",
    "explanation": "答案解析",
    "concept": "相关概念",
    "difficulty": "难度级别"
  }},
  ...其他题目
]

注意：
- 确保题目难度分布合理，基础、中等、高级难度的题目都要有
- 确保题目类型分布均匀
- 确保题目内容准确，符合学科特点
- 解析应该清晰解释为什么答案是正确的
"""
            
            # 调用LLM API
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一位专业的教育测评专家，擅长根据教材内容设计高质量的测验题目。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=2000
            )
            
            # 解析响应
            result = json.loads(response.choices[0].message.content)
            
            # 确保返回的是列表
            if isinstance(result, dict) and "quizzes" in result:
                quizzes = result["quizzes"]
            elif isinstance(result, list):
                quizzes = result
            else:
                logger.warning("LLM返回的结果格式不正确，将使用模板生成题目")
                return self._generate_template_quizzes(
                    important_concepts, 
                    content_analysis.get('chapters', []), 
                    subject, 
                    content_analysis.get('difficulty_level', 'intermediate'), 
                    num_questions
                )
            
            # 处理结果
            processed_quizzes = []
            for quiz in quizzes:
                # 确保每个题目包含必要的字段
                if "question" in quiz and "type" in quiz:
                    # 添加题目ID
                    quiz["id"] = f"q_{len(processed_quizzes) + 1}"
                    processed_quizzes.append(quiz)
            
            # 如果生成的题目少于要求的数量，使用模板补充
            if len(processed_quizzes) < num_questions:
                template_quizzes = self._generate_template_quizzes(
                    important_concepts, 
                    content_analysis.get('chapters', []), 
                    subject, 
                    content_analysis.get('difficulty_level', 'intermediate'), 
                    num_questions - len(processed_quizzes)
                )
                processed_quizzes.extend(template_quizzes)
            
            return processed_quizzes
            
        except Exception as e:
            logger.error(f"LLM生成题目失败: {e}")
            return self._generate_template_quizzes(
                content_analysis.get('important_concepts', []), 
                content_analysis.get('chapters', []), 
                content_analysis.get('subject', 'general'), 
                content_analysis.get('difficulty_level', 'intermediate'), 
                num_questions
            )
    
    def _generate_template_quizzes(self, 
                                  important_concepts: List[Dict[str, Any]],
                                  chapters: List[Dict[str, Any]],
                                  subject: str,
                                  global_difficulty: str,
                                  num_questions: int) -> List[Dict[str, Any]]:
        """使用模板生成测验题目"""
        quizzes = []
        
        # 确定各种题型的数量
        question_types = ['multiple_choice', 'true_false', 'fill_blank', 'short_answer', 'definition']
        type_counts = {}
        remaining = num_questions
        
        # 均匀分配各题型的数量
        base_count = num_questions // len(question_types)
        for q_type in question_types:
            type_counts[q_type] = base_count
            remaining -= base_count
        
        # 分配剩余的题目
        for i in range(remaining):
            type_counts[question_types[i % len(question_types)]] += 1
        
        # 为每个概念生成题目
        concepts_used = []
        
        # 按重要性排序概念
        sorted_concepts = sorted(
            important_concepts, 
            key=lambda x: x.get('importance', 0), 
            reverse=True
        )
        
        # 从章节中提取额外概念
        chapter_concepts = []
        for chapter in chapters:
            chapter_concepts.extend(chapter.get('key_concepts', []))
        
        # 如果重要概念不足，使用章节概念补充
        all_concept_names = [c.get('concept', '') for c in sorted_concepts]
        for concept in chapter_concepts:
            if concept not in all_concept_names:
                # 将字符串概念转换为字典格式
                sorted_concepts.append({
                    'concept': concept,
                    'importance': 1,
                    'chapters': []
                })
        
        # 生成各类型题目
        quiz_id = 1
        for q_type, count in type_counts.items():
            for _ in range(count):
                # 如果所有概念都已使用过但题目数量不足，重新开始使用概念
                if not sorted_concepts or len(concepts_used) >= len(sorted_concepts):
                    concepts_used = []
                
                # 选择下一个未使用的概念
                concept_data = None
                for c in sorted_concepts:
                    if c.get('concept', '') not in concepts_used:
                        concept_data = c
                        concepts_used.append(c.get('concept', ''))
                        break
                
                # 如果没有找到新概念，使用随机概念
                if concept_data is None and sorted_concepts:
                    concept_data = random.choice(sorted_concepts)
                
                # 如果没有任何概念，使用默认值
                if concept_data is None:
                    concept_data = {'concept': '课程内容', 'importance': 1, 'chapters': []}
                
                # 生成题目
                quiz = self._generate_quiz_from_template(
                    q_type, 
                    concept_data, 
                    subject, 
                    global_difficulty
                )
                
                # 添加ID
                quiz['id'] = f"q_{quiz_id}"
                quiz_id += 1
                
                quizzes.append(quiz)
        
        return quizzes
    
    def _generate_quiz_from_template(self, 
                                    quiz_type: str, 
                                    concept_data: Dict[str, Any],
                                    subject: str,
                                    global_difficulty: str) -> Dict[str, Any]:
        """根据模板生成特定类型的题目"""
        concept = concept_data.get('concept', '概念')
        
        # 随机选择难度级别，但倾向于全局难度
        difficulties = ['beginner', 'intermediate', 'advanced']
        weights = [0.2, 0.2, 0.2]  # 基础权重
        
        # 增加全局难度的权重
        if global_difficulty in difficulties:
            index = difficulties.index(global_difficulty)
            weights[index] += 0.4
        
        difficulty = random.choices(difficulties, weights=weights)[0]
        
        # 根据题型生成题目
        if quiz_type == 'multiple_choice':
            return self._generate_multiple_choice(concept, subject, difficulty)
        elif quiz_type == 'true_false':
            return self._generate_true_false(concept, subject, difficulty)
        elif quiz_type == 'fill_blank':
            return self._generate_fill_blank(concept, subject, difficulty)
        elif quiz_type == 'short_answer':
            return self._generate_short_answer(concept, subject, difficulty)
        elif quiz_type == 'definition':
            return self._generate_definition(concept, subject, difficulty)
        else:
            # 默认生成多选题
            return self._generate_multiple_choice(concept, subject, difficulty)
    
    def _generate_multiple_choice(self, concept: str, subject: str, difficulty: str) -> Dict[str, Any]:
        """生成多选题"""
        # 选择模板
        template = random.choice(self.quiz_templates['multiple_choice'])
        
        # 生成问题
        question = template.format(concept=concept)
        
        # 生成选项 (这里使用简化的实现)
        options = [
            f"{concept}是{subject}中的重要概念，它描述了特定的现象或规律。",
            f"{concept}通常与{self._get_related_term(subject)}相关联。",
            f"{concept}是一个复杂的概念，需要结合实际例子理解。",
            f"{concept}的应用范围非常广泛，几乎涵盖{subject}的所有领域。"
        ]
        
        # 随机选择正确答案
        correct_index = random.randint(0, len(options) - 1)
        
        # 返回多选题
        return {
            'type': 'multiple_choice',
            'question': question,
            'options': options,
            'answer': correct_index,
            'explanation': f"选项{chr(65+correct_index)}正确描述了{concept}的核心特点。",
            'concept': concept,
            'difficulty': difficulty
        }
    
    def _generate_true_false(self, concept: str, subject: str, difficulty: str) -> Dict[str, Any]:
        """生成判断题"""
        # 生成陈述
        statements = [
            f"{concept}是{subject}中的核心概念之一",
            f"{concept}与{self._get_related_term(subject)}没有直接关系",
            f"在大多数情况下，{concept}可以简化为更基本的概念",
            f"深入理解{concept}需要掌握{self._get_related_term(subject)}的基础知识"
        ]
        
        statement = random.choice(statements)
        
        # 随机决定正确与否
        is_correct = random.choice([True, False])
        
        # 如果设置为错误，修改陈述
        if not is_correct:
            negations = [
                f"{concept}不是{subject}中的重要概念",
                f"{concept}只在特殊情况下适用",
                f"{concept}已经被现代{subject}理论所淘汰",
                f"研究表明，{concept}的传统解释是不准确的"
            ]
            statement = random.choice(negations)
        
        # 选择模板
        template = random.choice(self.quiz_templates['true_false'])
        
        # 生成问题
        question = template.format(statement=statement)
        
        # 返回判断题
        return {
            'type': 'true_false',
            'question': question,
            'answer': is_correct,
            'explanation': f"这个陈述{'正确' if is_correct else '不正确'}，因为{concept}{'是' if is_correct else '不是'}{subject}中的重要概念。",
            'concept': concept,
            'difficulty': difficulty
        }
    
    def _generate_fill_blank(self, concept: str, subject: str, difficulty: str) -> Dict[str, Any]:
        """生成填空题"""
        # 生成包含空白的陈述
        statements_with_blank = [
            f"在{subject}中，___被定义为{self._get_simplified_definition(concept)}。",
            f"研究{concept}的主要目的是___。",
            f"___是{concept}的关键特性之一。",
            f"{concept}与{self._get_related_term(subject)}的关系可以描述为___。"
        ]
        
        statement = random.choice(statements_with_blank)
        
        # 根据空白位置确定答案
        answers = [
            f"{concept}",
            f"理解{subject}中的基本规律和现象",
            f"可测量性和可重复性",
            f"相互补充和依赖"
        ]
        
        # 选择与陈述对应的答案
        answer_index = statements_with_blank.index(statement)
        answer = answers[answer_index]
        
        # 选择模板
        template = random.choice(self.quiz_templates['fill_blank'])
        
        # 生成问题
        if "{statement_with_blank}" in template:
            question = template.format(statement_with_blank=statement)
        else:
            question = template.format(concept=concept, context=subject)
        
        # 返回填空题
        return {
            'type': 'fill_blank',
            'question': question,
            'answer': answer,
            'explanation': f"填入"{answer}"后，这个陈述准确描述了{concept}的特性或应用。",
            'concept': concept,
            'difficulty': difficulty
        }
    
    def _generate_short_answer(self, concept: str, subject: str, difficulty: str) -> Dict[str, Any]:
        """生成简答题"""
        # 选择模板
        template = random.choice(self.quiz_templates['short_answer'])
        
        # 生成相关概念
        related_concept = self._get_related_term(subject)
        
        # 生成问题
        question = template.format(concept=concept, related_concept=related_concept)
        
        # 生成参考答案
        answer = f"{concept}指的是{self._get_simplified_definition(concept)}。它在{subject}中非常重要，因为它帮助我们理解{related_concept}等相关概念。"
        
        # 返回简答题
        return {
            'type': 'short_answer',
            'question': question,
            'answer': answer,
            'explanation': f"一个好的回答应该包含{concept}的定义、重要性以及与{related_concept}的关系。",
            'concept': concept,
            'difficulty': difficulty
        }
    
    def _generate_definition(self, concept: str, subject: str, difficulty: str) -> Dict[str, Any]:
        """生成定义题"""
        # 选择模板
        template = random.choice(self.quiz_templates['definition'])
        
        # 生成问题
        question = template.format(concept=concept, subject=subject)
        
        # 生成答案
        answer = self._get_simplified_definition(concept)
        
        # 返回定义题
        return {
            'type': 'definition',
            'question': question,
            'answer': answer,
            'explanation': f"这是{concept}在{subject}领域中的标准定义。理解这个定义对掌握整个概念至关重要。",
            'concept': concept,
            'difficulty': difficulty
        }
    
    def _get_related_term(self, subject: str) -> str:
        """获取与学科相关的术语"""
        related_terms = {
            'physics': ["力", "能量", "动量", "电场", "磁场", "波动", "热力学"],
            'chemistry': ["元素", "分子", "化合物", "反应", "催化剂", "电解质"],
            'biology': ["细胞", "基因", "蛋白质", "酶", "DNA", "进化", "生态系统"],
            'math': ["函数", "极限", "导数", "积分", "方程", "矩阵", "向量"]
        }
        
        subject_terms = related_terms.get(subject, ["基本概念", "理论基础", "应用领域", "研究方法"])
        return random.choice(subject_terms)
    
    def _get_simplified_definition(self, concept: str) -> str:
        """生成概念的简化定义"""
        # 这里使用非常简化的实现，实际应用中应该使用更复杂的逻辑或数据库
        definitions = [
            f"描述特定现象或过程的专业术语",
            f"用于解释特定现象的理论框架的一部分",
            f"在特定条件下观察到的现象或规律",
            f"用于分析和预测系统行为的基本单位"
        ]
        
        return f"{concept}是{random.choice(definitions)}"

class QuizRenderer:
    """测验渲染器，用于将测验题目转换为HTML或其他格式"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 样式配置
        self.style_config = self._load_style_config()
    
    def _load_style_config(self) -> Dict[str, Any]:
        """加载样式配置"""
        default_config = {
            'theme': 'light',
            'font_family': 'Arial, sans-serif',
            'primary_color': '#4a86e8',
            'secondary_color': '#f1c232',
            'background_color': '#ffffff',
            'text_color': '#333333',
            'border_radius': '5px',
            'animation': True
        }
        
        # 合并用户配置和默认配置
        user_config = self.config.get('style', {})
        for key, default_value in default_config.items():
            if key not in user_config:
                user_config[key] = default_value
        
        return user_config
    
    def render_quiz_html(self, quizzes: List[Dict[str, Any]], title: str = "知识检测") -> str:
        """
        将测验题目渲染为HTML
        
        Args:
            quizzes: 测验题目列表
            title: 测验标题
            
        Returns:
            HTML字符串
        """
        # 创建CSS样式
        css = self._generate_css()
        
        # 创建JavaScript
        js = self._generate_js()
        
        # 创建HTML头部
        html_head = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
            {css}
            </style>
        </head>
        <body>
        """
        
        # 创建HTML主体
        html_body = f"""
        <div class="quiz-container">
            <div class="quiz-header">
                <h1>{title}</h1>
                <div class="quiz-progress">
                    <span id="current-question">1</span> / <span id="total-questions">{len(quizzes)}</span>
                </div>
            </div>
            
            <div class="quiz-content">
        """
        
        # 添加每个题目
        for i, quiz in enumerate(quizzes):
            quiz_html = self._render_quiz_item(quiz, i == 0)
            html_body += quiz_html
        
        # 添加结果区域
        html_body += """
            </div>
            
            <div class="quiz-result" id="quiz-result" style="display: none;">
                <h2>测验结果</h2>
                <div class="result-summary">
                    <p>总分: <span id="total-score">0</span> / <span id="max-score">0</span></p>
                    <p>正确率: <span id="accuracy">0%</span></p>
                </div>
                <div class="result-details" id="result-details">
                    <!-- 结果详情将通过JavaScript填充 -->
                </div>
                <button class="quiz-button restart-button" onclick="restartQuiz()">重新开始</button>
            </div>
            
            <div class="quiz-controls" id="quiz-controls">
                <button class="quiz-button" id="prev-button" onclick="prevQuestion()" disabled>上一题</button>
                <button class="quiz-button" id="next-button" onclick="nextQuestion()">下一题</button>
                <button class="quiz-button submit-button" id="submit-button" style="display: none;" onclick="submitQuiz()">提交测验</button>
            </div>
        </div>
        """
        
        # 添加JavaScript
        html_foot = f"""
        <script>
        {js}
        </script>
        </body>
        </html>
        """
        
        # 合并所有部分
        html = html_head + html_body + html_foot
        
        return html
    
    def _render_quiz_item(self, quiz: Dict[str, Any], is_visible: bool) -> str:
        """渲染单个测验题目"""
        quiz_id = quiz.get('id', 'q_unknown')
        question = quiz.get('question', '')
        quiz_type = quiz.get('type', '')
        
        # 设置可见性
        display_style = "block" if is_visible else "none"
        
        # 题目开始
        html = f"""
        <div class="quiz-question" id="{quiz_id}" style="display: {display_style};" data-type="{quiz_type}">
            <div class="question-header">
                <div class="question-type">{self._get_question_type_text(quiz_type)}</div>
                <div class="question-difficulty">{self._get_difficulty_stars(quiz.get('difficulty', 'intermediate'))}</div>
            </div>
            <div class="question-text">{question}</div>
        """
        
        # 根据题型添加不同的内容
        if quiz_type == 'multiple_choice':
            html += self._render_multiple_choice(quiz)
        elif quiz_type == 'true_false':
            html += self._render_true_false(quiz)
        elif quiz_type == 'fill_blank':
            html += self._render_fill_blank(quiz)
        elif quiz_type in ['short_answer', 'definition']:
            html += self._render_text_answer(quiz)
        
        # 添加反馈区域
        html += """
            <div class="question-feedback" style="display: none;">
                <div class="feedback-content"></div>
                <div class="explanation"></div>
            </div>
        </div>
        """
        
        return html
    
    def _render_multiple_choice(self, quiz: Dict[str, Any]) -> str:
        """渲染多选题"""
        options = quiz.get('options', [])
        quiz_id = quiz.get('id', 'q_unknown')
        
        html = """<div class="question-options">"""
        
        for i, option in enumerate(options):
            option_id = f"{quiz_id}_option_{i}"
            html += f"""
            <div class="option">
                <input type="radio" id="{option_id}" name="{quiz_id}_options" value="{i}">
                <label for="{option_id}">{option}</label>
            </div>
            """
        
        html += """</div>"""
        
        return html
    
    def _render_true_false(self, quiz: Dict[str, Any]) -> str:
        """渲染判断题"""
        quiz_id = quiz.get('id', 'q_unknown')
        
        html = f"""
        <div class="question-options true-false">
            <div class="option">
                <input type="radio" id="{quiz_id}_true" name="{quiz_id}_options" value="true">
                <label for="{quiz_id}_true">正确</label>
            </div>
            <div class="option">
                <input type="radio" id="{quiz_id}_false" name="{quiz_id}_options" value="false">
                <label for="{quiz_id}_false">错误</label>
            </div>
        </div>
        """
        
        return html
    
    def _render_fill_blank(self, quiz: Dict[str, Any]) -> str:
        """渲染填空题"""
        quiz_id = quiz.get('id', 'q_unknown')
        
        html = f"""
        <div class="question-fill-blank">
            <input type="text" id="{quiz_id}_answer" class="fill-blank-input" placeholder="请在此输入答案">
        </div>
        """
        
        return html
    
    def _render_text_answer(self, quiz: Dict[str, Any]) -> str:
        """渲染文本答案题（简答题、定义题）"""
        quiz_id = quiz.get('id', 'q_unknown')
        
        html = f"""
        <div class="question-text-answer">
            <textarea id="{quiz_id}_answer" class="text-answer-input" rows="4" placeholder="请在此输入你的回答"></textarea>
        </div>
        """
        
        return html
    
    def _get_question_type_text(self, quiz_type: str) -> str:
        """获取题型的中文名称"""
        type_map = {
            'multiple_choice': '选择题',
            'true_false': '判断题',
            'fill_blank': '填空题',
            'short_answer': '简答题',
            'definition': '定义题'
        }
        
        return type_map.get(quiz_type, '未知题型')
    
    def _get_difficulty_stars(self, difficulty: str) -> str:
        """根据难度级别返回星星表示"""
        if difficulty == 'beginner':
            return '★☆☆'
        elif difficulty == 'intermediate':
            return '★★☆'
        elif difficulty == 'advanced':
            return '★★★'
        else:
            return '★★☆'  # 默认中等难度
    
    def _generate_css(self) -> str:
        """生成CSS样式"""
        style = self.style_config
        
        css = f"""
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: {style['font_family']};
            background-color: {style['background_color']};
            color: {style['text_color']};
            line-height: 1.6;
            padding: 20px;
        }}
        
        .quiz-container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            border-radius: {style['border_radius']};
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }}
        
        .quiz-header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }}
        
        .quiz-header h1 {{
            color: {style['primary_color']};
            margin-bottom: 10px;
        }}
        
        .quiz-progress {{
            font-size: 16px;
            color: #666;
        }}
        
        .quiz-content {{
            margin-bottom: 30px;
        }}
        
        .quiz-question {{
            margin-bottom: 20px;
        }}
        
        .question-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        
        .question-type {{
            background-color: {style['primary_color']};
            color: white;
            padding: 5px 10px;
            border-radius: {style['border_radius']};
            font-size: 14px;
        }}
        
        .question-difficulty {{
            color: {style['secondary_color']};
            font-weight: bold;
        }}
        
        .question-text {{
            font-size: 18px;
            margin-bottom: 15px;
            line-height: 1.4;
        }}
        
        .question-options {{
            margin-bottom: 15px;
        }}
        
        .option {{
            margin-bottom: 10px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: {style['border_radius']};
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        
        .option:hover {{
            background-color: #f9f9f9;
        }}
        
        .option input {{
            margin-right: 10px;
        }}
        
        .option input {
            margin-right: 10px;
        }
        
        .true-false {
            display: flex;
            gap: 15px;
        }
        
        .true-false .option {
            flex: 1;
            text-align: center;
        }
        
        .fill-blank-input,
        .text-answer-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: {style['border_radius']};
            font-size: 16px;
        }
        
        .text-answer-input {
            resize: vertical;
        }
        
        .question-feedback {
            margin-top: 15px;
            padding: 15px;
            border-radius: {style['border_radius']};
            background-color: #f9f9f9;
        }
        
        .feedback-content {
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .feedback-content.correct {
            color: #4caf50;
        }
        
        .feedback-content.incorrect {
            color: #f44336;
        }
        
        .explanation {
            font-size: 14px;
            color: #666;
        }
        
        .quiz-controls {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        
        .quiz-button {
            padding: 10px 20px;
            border: none;
            border-radius: {style['border_radius']};
            background-color: {style['primary_color']};
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .quiz-button:hover {
            background-color: {self._adjust_color_brightness(style['primary_color'], -20)};
        }
        
        .quiz-button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        
        .submit-button {
            background-color: {style['secondary_color']};
        }
        
        .submit-button:hover {
            background-color: {self._adjust_color_brightness(style['secondary_color'], -20)};
        }
        
        .restart-button {
            background-color: #4caf50;
            margin: 20px auto;
            display: block;
        }
        
        .restart-button:hover {
            background-color: #45a049;
        }
        
        .quiz-result {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .quiz-result h2 {
            color: {style['primary_color']};
            margin-bottom: 20px;
        }
        
        .result-summary {
            margin-bottom: 20px;
            font-size: 18px;
        }
        
        .result-details {
            margin-top: 30px;
            text-align: left;
        }
        
        .result-item {
            margin-bottom: 15px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: {style['border_radius']};
        }
        
        .result-item.correct {
            border-left: 4px solid #4caf50;
        }
        
        .result-item.incorrect {
            border-left: 4px solid #f44336;
        }
        
        .result-question {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .result-answer {
            margin-bottom: 5px;
        }
        
        .result-correct-answer {
            color: #4caf50;
            font-weight: bold;
        }
        
        @media (max-width: 600px) {
            .quiz-container {
                padding: 15px;
            }
            
            .question-text {
                font-size: 16px;
            }
            
            .quiz-button {
                padding: 8px 15px;
                font-size: 14px;
            }
        }
        """
        
        return css
    
    def _generate_js(self) -> str:
        """生成JavaScript代码"""
        js = """
        // 测验数据
        const quizData = {
            currentQuestion: 0,
            questions: [],
            answers: {},
            result: {
                score: 0,
                maxScore: 0,
                details: []
            }
        };
        
        // 初始化测验
        function initQuiz() {
            // 获取所有问题
            const questionElements = document.querySelectorAll('.quiz-question');
            questionElements.forEach((element, index) => {
                const questionId = element.id;
                const questionType = element.dataset.type;
                
                quizData.questions.push({
                    id: questionId,
                    type: questionType,
                    index: index
                });
            });
            
            // 更新总题数
            document.getElementById('total-questions').textContent = quizData.questions.length;
            
            // 初始化第一题
            showQuestion(0);
        }
        
        // 显示特定问题
        function showQuestion(index) {
            // 隐藏所有问题
            const questionElements = document.querySelectorAll('.quiz-question');
            questionElements.forEach(element => {
                element.style.display = 'none';
            });
            
            // 显示当前问题
            const currentQuestion = quizData.questions[index];
            document.getElementById(currentQuestion.id).style.display = 'block';
            
            // 更新当前问题索引
            quizData.currentQuestion = index;
            document.getElementById('current-question').textContent = index + 1;
            
            // 更新按钮状态
            updateButtonsState();
        }
        
        // 更新按钮状态
        function updateButtonsState() {
            const prevButton = document.getElementById('prev-button');
            const nextButton = document.getElementById('next-button');
            const submitButton = document.getElementById('submit-button');
            
            // 上一题按钮
            prevButton.disabled = quizData.currentQuestion === 0;
            
            // 下一题和提交按钮
            if (quizData.currentQuestion === quizData.questions.length - 1) {
                nextButton.style.display = 'none';
                submitButton.style.display = 'block';
            } else {
                nextButton.style.display = 'block';
                submitButton.style.display = 'none';
            }
        }
        
        // 上一题
        function prevQuestion() {
            if (quizData.currentQuestion > 0) {
                // 保存当前答案
                saveCurrentAnswer();
                
                // 显示上一题
                showQuestion(quizData.currentQuestion - 1);
            }
        }
        
        // 下一题
        function nextQuestion() {
            if (quizData.currentQuestion < quizData.questions.length - 1) {
                // 保存当前答案
                saveCurrentAnswer();
                
                // 显示下一题
                showQuestion(quizData.currentQuestion + 1);
            }
        }
        
        // 保存当前答案
        function saveCurrentAnswer() {
            const currentQuestion = quizData.questions[quizData.currentQuestion];
            const questionId = currentQuestion.id;
            const questionType = currentQuestion.type;
            
            let answer = null;
            
            // 根据题型获取答案
            if (questionType === 'multiple_choice') {
                const selectedOption = document.querySelector(`input[name="${questionId}_options"]:checked`);
                if (selectedOption) {
                    answer = parseInt(selectedOption.value);
                }
            } else if (questionType === 'true_false') {
                const selectedOption = document.querySelector(`input[name="${questionId}_options"]:checked`);
                if (selectedOption) {
                    answer = selectedOption.value === 'true';
                }
            } else if (questionType === 'fill_blank' || questionType === 'short_answer' || questionType === 'definition') {
                const inputElement = document.getElementById(`${questionId}_answer`);
                if (inputElement && inputElement.value.trim() !== '') {
                    answer = inputElement.value.trim();
                }
            }
            
            // 保存答案
            if (answer !== null) {
                quizData.answers[questionId] = answer;
            }
        }
        
        // 提交测验
        function submitQuiz() {
            // 保存最后一题的答案
            saveCurrentAnswer();
            
            // 获取正确答案并计算分数
            const quizResult = calculateResult();
            
            // 显示结果
            displayResult(quizResult);
        }
        
        // 计算测验结果
        function calculateResult() {
            // 这里应该从服务器获取正确答案
            // 在前端演示中，我们使用模拟数据
            
            const mockAnswers = {
                // 每个题目的正确答案将在实际使用时从服务器获取
            };
            
            let score = 0;
            let maxScore = quizData.questions.length;
            let details = [];
            
            // 评估每个问题
            quizData.questions.forEach(question => {
                const userAnswer = quizData.answers[question.id];
                const correctAnswer = mockAnswers[question.id] || 0; // 默认第一个选项为正确答案
                
                let isCorrect = false;
                
                // 根据题型判断答案是否正确
                if (question.type === 'multiple_choice') {
                    isCorrect = userAnswer === correctAnswer;
                } else if (question.type === 'true_false') {
                    isCorrect = userAnswer === correctAnswer;
                } else if (question.type === 'fill_blank') {
                    // 简单比较（实际应用中可能需要更复杂的比较逻辑）
                    isCorrect = userAnswer && userAnswer.toLowerCase() === correctAnswer.toLowerCase();
                } else if (question.type === 'short_answer' || question.type === 'definition') {
                    // 简答题和定义题通常需要人工评分
                    // 这里简单实现自动评分（关键词匹配）
                    isCorrect = userAnswer && correctAnswer && userAnswer.toLowerCase().includes(correctAnswer.toLowerCase());
                }
                
                // 如果答案正确，增加分数
                if (isCorrect) {
                    score++;
                }
                
                // 添加到详情
                details.push({
                    questionId: question.id,
                    userAnswer: userAnswer,
                    correctAnswer: correctAnswer,
                    isCorrect: isCorrect
                });
            });
            
            // 保存结果
            quizData.result = {
                score: score,
                maxScore: maxScore,
                details: details
            };
            
            return quizData.result;
        }
        
        // 显示测验结果
        function displayResult(result) {
            // 隐藏测验内容和控制按钮
            document.querySelector('.quiz-content').style.display = 'none';
            document.getElementById('quiz-controls').style.display = 'none';
            
            // 显示结果区域
            const resultElement = document.getElementById('quiz-result');
            resultElement.style.display = 'block';
            
            // 更新分数
            document.getElementById('total-score').textContent = result.score;
            document.getElementById('max-score').textContent = result.maxScore;
            
            // 计算正确率
            const accuracy = (result.score / result.maxScore) * 100;
            document.getElementById('accuracy').textContent = accuracy.toFixed(1) + '%';
            
            // 显示详细结果
            const detailsElement = document.getElementById('result-details');
            detailsElement.innerHTML = '';
            
            result.details.forEach((detail, index) => {
                const question = document.getElementById(detail.questionId);
                const questionText = question.querySelector('.question-text').textContent;
                
                const detailHtml = `
                    <div class="result-item ${detail.isCorrect ? 'correct' : 'incorrect'}">
                        <div class="result-question">${index + 1}. ${questionText}</div>
                        <div class="result-answer">你的答案: ${formatAnswer(detail.userAnswer, question.dataset.type)}</div>
                        ${!detail.isCorrect ? `<div class="result-correct-answer">正确答案: ${formatAnswer(detail.correctAnswer, question.dataset.type)}</div>` : ''}
                    </div>
                `;
                
                detailsElement.innerHTML += detailHtml;
            });
        }
        
        // 格式化答案显示
        function formatAnswer(answer, type) {
            if (answer === undefined || answer === null) {
                return '未作答';
            }
            
            if (type === 'multiple_choice') {
                // 将选项索引转换为选项显示文本
                const options = ['A', 'B', 'C', 'D'];
                return options[answer] || answer;
            } else if (type === 'true_false') {
                return answer ? '正确' : '错误';
            } else {
                return answer;
            }
        }
        
        // 重新开始测验
        function restartQuiz() {
            // 重置数据
            quizData.currentQuestion = 0;
            quizData.answers = {};
            quizData.result = {
                score: 0,
                maxScore: 0,
                details: []
            };
            
            // 重置界面
            document.querySelector('.quiz-content').style.display = 'block';
            document.getElementById('quiz-controls').style.display = 'flex';
            document.getElementById('quiz-result').style.display = 'none';
            
            // 清除所有选择和输入
            const radioInputs = document.querySelectorAll('input[type="radio"]');
            radioInputs.forEach(input => {
                input.checked = false;
            });
            
            const textInputs = document.querySelectorAll('input[type="text"], textarea');
            textInputs.forEach(input => {
                input.value = '';
            });
            
            // 隐藏所有反馈
            const feedbackElements = document.querySelectorAll('.question-feedback');
            feedbackElements.forEach(element => {
                element.style.display = 'none';
            });
            
            // 显示第一题
            showQuestion(0);
        }
        
        // 页面加载完成后初始化测验
        document.addEventListener('DOMContentLoaded', initQuiz);
        """
        
        return js
    
    def _adjust_color_brightness(self, color: str, amount: int) -> str:
        """调整颜色亮度"""
        # 如果颜色是十六进制格式(#RRGGBB)
        if color.startswith('#'):
            # 移除#符号
            color = color[1:]
            
            # 将十六进制转换为RGB
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            
            # 调整亮度
            r = max(0, min(255, r + amount))
            g = max(0, min(255, g + amount))
            b = max(0, min(255, b + amount))
            
            # 转换回十六进制
            return f"#{r:02x}{g:02x}{b:02x}"
        
        return color

class QuizExporter:
    """测验导出器，用于将测验以不同格式导出"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def export_to_html(self, quizzes: List[Dict[str, Any]], output_path: str, title: str = "知识检测") -> str:
        """
        将测验导出为HTML文件
        
        Args:
            quizzes: 测验题目列表
            output_path: 输出文件路径
            title: 测验标题
            
        Returns:
            输出文件路径
        """
        # 使用渲染器生成HTML
        renderer = QuizRenderer(self.config)
        html_content = renderer.render_quiz_html(quizzes, title)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML测验已生成: {output_path}")
        return output_path
    
    def export_to_json(self, quizzes: List[Dict[str, Any]], output_path: str) -> str:
        """
        将测验导出为JSON文件
        
        Args:
            quizzes: 测验题目列表
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        # 写入JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(quizzes, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON测验已生成: {output_path}")
        return output_path
    
    def export_to_text(self, quizzes: List[Dict[str, Any]], output_path: str, title: str = "知识检测") -> str:
        """
        将测验导出为纯文本文件
        
        Args:
            quizzes: 测验题目列表
            output_path: 输出文件路径
            title: 测验标题
            
        Returns:
            输出文件路径
        """
        # 生成文本内容
        text_content = f"{title}\n{'=' * len(title)}\n\n"
        
        for i, quiz in enumerate(quizzes):
            question_type = self._get_question_type_text(quiz.get('type', ''))
            difficulty = self._get_difficulty_text(quiz.get('difficulty', 'intermediate'))
            
            text_content += f"{i+1}. [{question_type} - {difficulty}] {quiz.get('question', '')}\n"
            
            # 根据题型添加不同的内容
            if quiz.get('type') == 'multiple_choice':
                options = quiz.get('options', [])
                for j, option in enumerate(options):
                    text_content += f"   {chr(65+j)}. {option}\n"
            elif quiz.get('type') == 'true_false':
                text_content += "   A. 正确\n   B. 错误\n"
            elif quiz.get('type') == 'fill_blank':
                text_content += "   (请在此填写答案)\n"
            elif quiz.get('type') in ['short_answer', 'definition']:
                text_content += "   (请在此作答)\n"
            
            # 添加答案和解析(通常不会在导出的测验中包含，这里仅为了完整性)
            if self.config.get('include_answers', False):
                text_content += f"\n   答案: {self._format_answer(quiz)}\n"
                text_content += f"   解析: {quiz.get('explanation', '')}\n"
            
            text_content += "\n\n"
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"文本测验已生成: {output_path}")
        return output_path
    
    def _get_question_type_text(self, quiz_type: str) -> str:
        """获取题型的中文名称"""
        type_map = {
            'multiple_choice': '选择题',
            'true_false': '判断题',
            'fill_blank': '填空题',
            'short_answer': '简答题',
            'definition': '定义题'
        }
        
        return type_map.get(quiz_type, '未知题型')
    
    def _get_difficulty_text(self, difficulty: str) -> str:
        """获取难度级别文本"""
        if difficulty == 'beginner':
            return '初级'
        elif difficulty == 'intermediate':
            return '中级'
        elif difficulty == 'advanced':
            return '高级'
        else:
            return '中级'
    
    def _format_answer(self, quiz: Dict[str, Any]) -> str:
        """格式化答案显示"""
        quiz_type = quiz.get('type', '')
        answer = quiz.get('answer')
        
        if quiz_type == 'multiple_choice':
            options = ['A', 'B', 'C', 'D']
            if isinstance(answer, int) and 0 <= answer < len(options):
                return options[answer]
            return str(answer)
        elif quiz_type == 'true_false':
            return '正确' if answer else '错误'
        else:
            return str(answer)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成交互式测验')
    parser.add_argument('content_file', help='内容分析结果JSON文件')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--format', '-f', choices=['html', 'json', 'text'], default='html', help='输出格式')
    parser.add_argument('--num_questions', '-n', type=int, default=10, help='生成题目数量')
    parser.add_argument('--title', '-t', default='知识检测', help='测验标题')
    parser.add_argument('--openai_api_key', help='OpenAI API密钥')
    parser.add_argument('--include_answers', action='store_true', help='在文本导出中包含答案')
    
    args = parser.parse_args()
    
    # 加载内容分析结果
    with open(args.content_file, 'r', encoding='utf-8') as f:
        content_analysis = json.load(f)
    
    # 配置
    config = {}
    if args.openai_api_key:
        config['llm_provider'] = 'openai'
        config['openai_api_key'] = args.openai_api_key
    
    if args.include_answers:
        config['include_answers'] = True
    
    # 设置默认输出路径
    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(args.content_file))[0]
        if args.format == 'html':
            output_path = f"{base_name}_quiz.html"
        elif args.format == 'json':
            output_path = f"{base_name}_quiz.json"
        else:
            output_path = f"{base_name}_quiz.txt"
    
    # 初始化测验生成器
    quiz_generator = QuizGenerator(config)
    
    # 生成测验题目
    quizzes = quiz_generator.generate_quizzes(content_analysis, args.num_questions)
    
    # 导出测验
    exporter = QuizExporter(config)
    
    if args.format == 'html':
        output_file = exporter.export_to_html(quizzes, output_path, args.title)
    elif args.format == 'json':
        output_file = exporter.export_to_json(quizzes, output_path)
    else:
        output_file = exporter.export_to_text(quizzes, output_path, args.title)
    
    print(f"测验已生成: {output_file}")
    print(f"共生成 {len(quizzes)} 道题目")

if __name__ == "__main__":
    main()