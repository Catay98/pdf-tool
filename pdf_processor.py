"""
PDF处理与结构化提取模块
---------------------
用于处理低质量翻印PDF，提取文本、识别结构并组织内容。
主要功能：
1. 图像预处理增强
2. OCR文本提取
3. 版面结构分析
4. 章节识别与分割
5. 内容清洗与重建
"""

import os
import io
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from paddleocr import PaddleOCR
import layoutparser as lp
import spacy
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import json
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF处理主类，协调各个子模块工作"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化PDF处理器
        
        Args:
            config: 配置参数，包含OCR设置、模型路径等
        """
        self.config = config or {}
        
        # 初始化OCR引擎
        self.tesseract_lang = self.config.get('tesseract_lang', 'chi_sim+eng')
        pytesseract.pytesseract.tesseract_cmd = self.config.get('tesseract_path', 'tesseract')
        
        # 初始化PaddleOCR (支持中文识别)
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=self.config.get('use_gpu', False))
        
        # 初始化版面分析模型
        self.layout_model = lp.Detectron2LayoutModel(
            config_path=self.config.get('layout_config', 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config'),
            model_path=self.config.get('layout_model', 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/model'),
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        
        # 初始化NLP模型用于文本处理
        self.nlp = spacy.load("zh_core_web_sm")
        
        # 初始化章节模式识别
        self.chapter_patterns = [
            r'^第[一二三四五六七八九十百零\d]+章[\s:：]*(.*?)$',
            r'^[0-9]+[\.\s]+(.*?)$',
            r'^Chapter\s+[0-9]+[\.\s]*(.*?)$'
        ]
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        处理PDF文件的主函数
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含PDF处理结果的字典，包括章节、段落、图表等信息
        """
        logger.info(f"开始处理PDF: {pdf_path}")
        
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"PDF共{total_pages}页")
        
        # 存储处理结果
        result = {
            "meta": {
                "filename": os.path.basename(pdf_path),
                "pages": total_pages,
                "processed_at": pd.Timestamp.now().isoformat()
            },
            "chapters": [],
            "current_chapter": {"title": "未命名章节", "content": [], "start_page": 0}
        }
        
        # 逐页处理
        for page_num in range(total_pages):
            logger.info(f"处理第{page_num+1}/{total_pages}页")
            
            # 处理单页并获取结果
            page_result = self._process_page(doc, page_num)
            
            # 检测章节标题并更新章节结构
            if page_result.get("chapter_title"):
                # 保存当前章节
                if result["current_chapter"]["content"]:
                    result["chapters"].append(result["current_chapter"])
                
                # 创建新章节
                result["current_chapter"] = {
                    "title": page_result["chapter_title"],
                    "content": [],
                    "start_page": page_num
                }
            
            # 将页面内容添加到当前章节
            result["current_chapter"]["content"].extend(page_result["content"])
        
        # 添加最后一个章节
        if result["current_chapter"]["content"]:
            result["chapters"].append(result["current_chapter"])
        
        # 关闭PDF
        doc.close()
        
        # 后处理：合并段落、清洗文本
        self._post_process_content(result)
        
        logger.info(f"PDF处理完成，共识别{len(result['chapters'])}个章节")
        return result
    
    def _process_page(self, doc: fitz.Document, page_num: int) -> Dict[str, Any]:
        """处理单个PDF页面"""
        page = doc[page_num]
        
        # 步骤1: 图像预处理
        pix = self._preprocess_page_image(page)
        
        # 步骤2: OCR文本提取
        ocr_results = self._perform_ocr(pix)
        
        # 步骤3: 版面分析
        layout_results = self._analyze_layout(pix, ocr_results)
        
        # 步骤4: 识别章节标题
        chapter_title = self._detect_chapter_title(layout_results)
        
        # 组织页面内容
        page_result = {
            "page_num": page_num,
            "chapter_title": chapter_title,
            "content": self._organize_page_content(layout_results)
        }
        
        return page_result
    
    def _preprocess_page_image(self, page: fitz.Page) -> np.ndarray:
        """
        预处理页面图像以提高OCR质量
        
        Args:
            page: PDF页面对象
            
        Returns:
            预处理后的页面图像
        """
        # 渲染页面为高分辨率图像
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # 转换为灰度图
        if img.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:  # RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 降噪
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # 提高对比度
        alpha = 1.5  # 对比度增强因子
        beta = 10    # 亮度增强
        enhanced = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
        
        return enhanced
    
    def _perform_ocr(self, img: np.ndarray) -> Dict[str, Any]:
        """
        执行OCR识别，融合多种OCR引擎结果
        
        Args:
            img: 预处理后的页面图像
            
        Returns:
            OCR识别结果
        """
        # 转换为PIL图像用于pytesseract
        pil_img = Image.fromarray(img)
        
        # Tesseract OCR
        try:
            tesseract_result = pytesseract.image_to_data(
                pil_img, 
                lang=self.tesseract_lang,
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            logger.warning(f"Tesseract OCR失败: {e}")
            tesseract_result = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}
        
        # PaddleOCR
        try:
            paddle_result = self.paddle_ocr.ocr(img, cls=True)
            paddle_texts = []
            paddle_boxes = []
            paddle_scores = []
            
            if paddle_result:
                for line in paddle_result:
                    for item in line:
                        box, (text, score) = item
                        paddle_texts.append(text)
                        paddle_boxes.append(box)
                        paddle_scores.append(score)
        except Exception as e:
            logger.warning(f"PaddleOCR失败: {e}")
            paddle_texts = []
            paddle_boxes = []
            paddle_scores = []
        
        # 合并结果 (简单实现，实际应考虑更复杂的融合策略)
        merged_result = {
            "tesseract": {
                "text": [text for text in tesseract_result["text"] if text.strip()],
                "conf": tesseract_result["conf"],
                "boxes": list(zip(
                    tesseract_result["left"],
                    tesseract_result["top"],
                    tesseract_result["width"],
                    tesseract_result["height"]
                ))
            },
            "paddle": {
                "text": paddle_texts,
                "conf": paddle_scores,
                "boxes": paddle_boxes
            }
        }
        
        # 返回融合后的结果
        return self._merge_ocr_results(merged_result)
    
    def _merge_ocr_results(self, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """融合多个OCR引擎的结果"""
        # 简单实现：如果Paddle识别出了文本，优先使用Paddle结果，否则使用Tesseract
        if ocr_results["paddle"]["text"]:
            primary_texts = ocr_results["paddle"]["text"]
            primary_boxes = ocr_results["paddle"]["boxes"]
        else:
            # 过滤出有意义的Tesseract结果
            valid_indices = [i for i, conf in enumerate(ocr_results["tesseract"]["conf"]) 
                            if conf > 30 and ocr_results["tesseract"]["text"][i].strip()]
            
            primary_texts = [ocr_results["tesseract"]["text"][i] for i in valid_indices]
            primary_boxes = [ocr_results["tesseract"]["boxes"][i] for i in valid_indices]
        
        # 排序文本块 (从上到下，从左到右)
        text_blocks = [(text, box) for text, box in zip(primary_texts, primary_boxes)]
        text_blocks.sort(key=lambda x: (x[1][1], x[1][0]) if isinstance(x[1][0], (int, float)) else 
                                       (x[1][0][1], x[1][0][0]))
        
        merged_result = {
            "text_blocks": [{"text": block[0], "box": block[1]} for block in text_blocks]
        }
        
        return merged_result
    
    def _analyze_layout(self, img: np.ndarray, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析页面版面结构
        
        Args:
            img: 页面图像
            ocr_results: OCR识别结果
            
        Returns:
            版面分析结果
        """
        # 将图像转换为RGB（版面分析模型需要）
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
            
        # 运行版面检测
        layout = self.layout_model.detect(img_rgb)
        
        # 处理检测结果
        structured_layout = []
        for block in layout:
            x_1, y_1, x_2, y_2 = block.coordinates
            block_type = block.type
            
            # 查找该区块内的文本
            block_texts = []
            for text_block in ocr_results["text_blocks"]:
                box = text_block["box"]
                
                # 获取文本块的边界
                if isinstance(box[0], (int, float)):
                    # Tesseract格式: [left, top, width, height]
                    txt_x1, txt_y1 = box[0], box[1]
                    txt_x2, txt_y2 = box[0] + box[2], box[1] + box[3]
                else:
                    # Paddle格式: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    txt_coords = np.array(box)
                    txt_x1, txt_y1 = txt_coords[:, 0].min(), txt_coords[:, 1].min()
                    txt_x2, txt_y2 = txt_coords[:, 0].max(), txt_coords[:, 1].max()
                
                # 检查文本是否在当前区块内
                if (txt_x1 >= x_1 and txt_x2 <= x_2 and 
                    txt_y1 >= y_1 and txt_y2 <= y_2):
                    block_texts.append(text_block["text"])
            
            # 添加到结构化版面
            structured_layout.append({
                "type": block_type,
                "coordinates": [x_1, y_1, x_2, y_2],
                "texts": block_texts
            })
            
        # 按照从上到下的顺序排列布局块
        structured_layout.sort(key=lambda x: x["coordinates"][1])
        
        return {
            "layout": structured_layout
        }
    
    def _detect_chapter_title(self, layout_results: Dict[str, Any]) -> Optional[str]:
        """检测页面中的章节标题"""
        # 查找所有标题类型的布局块
        title_blocks = [block for block in layout_results["layout"] 
                       if block["type"] == "Title" and block["texts"]]
        
        # 检查每个标题是否匹配章节模式
        for block in title_blocks:
            for text in block["texts"]:
                for pattern in self.chapter_patterns:
                    match = re.match(pattern, text)
                    if match:
                        # 如果有标题内容，提取它，否则使用整个匹配文本
                        title = match.group(1) if match.group(1) else text
                        return title.strip()
        
        # 如果没找到章节标题，看看普通文本块的开头是否匹配
        for block in layout_results["layout"]:
            if block["texts"] and block["coordinates"][1] < 100:  # 只检查页面顶部
                for text in block["texts"]:
                    for pattern in self.chapter_patterns:
                        match = re.match(pattern, text)
                        if match:
                            title = match.group(1) if match.group(1) else text
                            return title.strip()
        
        return None
    
    def _organize_page_content(self, layout_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据版面分析结果组织页面内容"""
        organized_content = []
        
        for block in layout_results["layout"]:
            block_type = block["type"]
            texts = block["texts"]
            
            if not texts:
                continue
                
            # 合并文本块中的文本
            text = " ".join(texts)
            
            # 创建内容项
            content_item = {
                "type": block_type.lower(),
                "text": text,
                "position": block["coordinates"]
            }
            
            organized_content.append(content_item)
        
        return organized_content
    
    def _post_process_content(self, result: Dict[str, Any]) -> None:
        """对提取的内容进行后处理，合并段落、清洗文本等"""
        for chapter in result["chapters"]:
            processed_content = []
            current_paragraph = ""
            current_type = None
            
            for item in chapter["content"]:
                item_type = item["type"]
                text = item["text"]
                
                # 清洗文本
                text = self._clean_text(text)
                
                if not text:
                    continue
                
                # 段落合并逻辑
                if item_type == "text":
                    if current_type == "text":
                        # 检查是否应该是同一段落
                        if not current_paragraph.endswith(("。", "！", "？", ".", "!", "?")):
                            current_paragraph += text
                        else:
                            # 结束当前段落并开始新段落
                            processed_content.append({
                                "type": "paragraph",
                                "text": current_paragraph
                            })
                            current_paragraph = text
                    else:
                        # 新段落
                        if current_type and current_paragraph:
                            processed_content.append({
                                "type": current_type,
                                "text": current_paragraph
                            })
                        current_paragraph = text
                        current_type = "paragraph"
                elif item_type == "title":
                    # 保存前一个内容
                    if current_type and current_paragraph:
                        processed_content.append({
                            "type": current_type,
                            "text": current_paragraph
                        })
                    
                    # 添加标题
                    processed_content.append({
                        "type": "subheading",
                        "text": text
                    })
                    
                    current_paragraph = ""
                    current_type = None
                else:
                    # 其他类型（如列表、表格等）
                    if current_type and current_paragraph:
                        processed_content.append({
                            "type": current_type,
                            "text": current_paragraph
                        })
                    
                    processed_content.append({
                        "type": item_type,
                        "text": text
                    })
                    
                    current_paragraph = ""
                    current_type = None
            
            # 处理最后一个内容
            if current_type and current_paragraph:
                processed_content.append({
                    "type": current_type,
                    "text": current_paragraph
                })
            
            # 更新章节内容
            chapter["content"] = processed_content
    
    def _clean_text(self, text: str) -> str:
        """清洗文本，修复OCR错误"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 修复常见OCR错误
        text = text.replace('0', 'O').replace('l', '1')
        
        # 修复标点符号
        text = text.replace(' ,', ',').replace(' .', '.')
        text = text.replace(' :', ':').replace(' ;', ';')
        
        # 使用NLP进行进一步清洗和修正
        doc = self.nlp(text)
        # 这里可以添加基于spaCy的更复杂文本修复逻辑
        
        return text

class ChapterExtractor:
    """章节提取器，用于识别和提取PDF中的章节结构"""
    
    def __init__(self):
        # 章节标题识别模式
        self.chapter_patterns = [
            r'^第[一二三四五六七八九十百零\d]+章[\s:：]*(.*?)$',
            r'^[0-9]+[\.\s]+(.*?)$',
            r'^Chapter\s+[0-9]+[\.\s]*(.*?)$'
        ]
    
    def extract_chapters(self, pdf_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从处理后的PDF内容中提取章节结构
        
        Args:
            pdf_content: PDFProcessor处理后的内容
            
        Returns:
            章节列表，每个章节包含标题、内容等信息
        """
        # 这里是一个简化版，实际实现会更复杂
        # 现在我们假设PDFProcessor已经初步识别了章节
        
        chapters = pdf_content.get("chapters", [])
        
        # 进一步处理每个章节
        for chapter in chapters:
            # 提取子章节
            subchapters = self._extract_subchapters(chapter["content"])
            if subchapters:
                chapter["subchapters"] = subchapters
                
        return chapters
    
    def _extract_subchapters(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从章节内容中提取子章节"""
        subchapters = []
        current_subchapter = None
        
        for item in content:
            if item["type"] == "subheading":
                # 保存前一个子章节
                if current_subchapter:
                    subchapters.append(current_subchapter)
                
                # 创建新子章节
                current_subchapter = {
                    "title": item["text"],
                    "content": []
                }
            elif current_subchapter:
                # 将内容添加到当前子章节
                current_subchapter["content"].append(item)
        
        # 添加最后一个子章节
        if current_subchapter:
            subchapters.append(current_subchapter)
            
        return subchapters

def main():
    """主函数示例"""
    # 配置参数
    config = {
        "tesseract_path": "D:\\Program Files\\Tesseract-OCR\\tesseract.exe",  # Windows 路径示例
        "tesseract_lang": "chi_sim+eng",
        "use_gpu": False
    }
    
    # 初始化处理器
    processor = PDFProcessor(config)
    
    # 处理PDF
    pdf_path = "example_textbook.pdf"
    result = processor.process_pdf(pdf_path)
    
    # 提取章节
    chapter_extractor = ChapterExtractor()
    chapters = chapter_extractor.extract_chapters(result)
    
    # 保存结果到JSON文件
    with open("processed_textbook.json", "w", encoding="utf-8") as f:
        json.dump(chapters, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到processed_textbook.json")

if __name__ == "__main__":
    import pandas as pd  # 用于时间戳
    main()