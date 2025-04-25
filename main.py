#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF知识点提炼工具 - 主程序
整合各个模块功能，提供命令行接口
"""

import os
import argparse
import logging
from datetime import datetime

# 导入自定义模块
from pdf_parser import PDFParser
from text_processor import TextProcessor
from knowledge_extractor import KnowledgeExtractor
from output_formatter import OutputFormatter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_knowledge_tool.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFKnowledgeTool:
    """PDF知识点提炼工具的主类，整合各个功能模块"""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.text_processor = TextProcessor()
        self.knowledge_extractor = KnowledgeExtractor()
        self.output_formatter = OutputFormatter()
    
    def process_pdf(self, pdf_path, output_format='markdown', extract_mode='auto', 
                   min_importance=0.5, output_dir=None):
        """
        处理单个PDF文件
        
        参数:
            pdf_path (str): PDF文件路径
            output_format (str): 输出格式 (markdown, json, txt)
            extract_mode (str): 提取模式 (auto, keywords, sentences, sections)
            min_importance (float): 最小重要性阈值 (0.0-1.0)
            output_dir (str): 输出目录
        
        返回:
            str: 输出文件路径
        """
        try:
            logger.info(f"开始处理PDF: {pdf_path}")
            
            # 1. 解析PDF
            pdf_text = self.pdf_parser.parse(pdf_path)
            logger.info(f"PDF解析完成，提取文本长度: {len(pdf_text)} 字符")
            
            # 2. 文本处理
            processed_text = self.text_processor.process(pdf_text)
            logger.info("文本预处理完成")
            
            # 3. 知识点提取
            knowledge_points = self.knowledge_extractor.extract(
                processed_text, 
                mode=extract_mode,
                min_importance=min_importance
            )
            logger.info(f"知识点提取完成，共 {len(knowledge_points)} 个知识点")
            
            # 4. 格式化输出
            if output_dir is None:
                output_dir = os.path.dirname(pdf_path)
                
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{filename}_knowledge_{timestamp}.{output_format.lower()}"
            output_path = os.path.join(output_dir, output_filename)
            
            self.output_formatter.format_and_save(
                knowledge_points,
                output_path,
                format_type=output_format,
                metadata={
                    'source_file': pdf_path,
                    'extraction_mode': extract_mode,
                    'min_importance': min_importance,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"输出已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"处理PDF时出错: {str(e)}", exc_info=True)
            raise
    
    def process_directory(self, dir_path, **kwargs):
        """
        处理目录中的所有PDF文件
        
        参数:
            dir_path (str): 包含PDF文件的目录路径
            **kwargs: 传递给process_pdf的参数
            
        返回:
            list: 输出文件路径列表
        """
        output_files = []
        try:
            logger.info(f"开始处理目录: {dir_path}")
            
            # 确保目录存在
            if not os.path.isdir(dir_path):
                raise ValueError(f"指定的路径不是一个目录: {dir_path}")
            
            # 查找所有PDF文件
            pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
            logger.info(f"找到 {len(pdf_files)} 个PDF文件")
            
            # 处理每个PDF文件
            for pdf_file in pdf_files:
                pdf_path = os.path.join(dir_path, pdf_file)
                output_file = self.process_pdf(pdf_path, **kwargs)
                output_files.append(output_file)
                
            logger.info(f"目录处理完成，输出 {len(output_files)} 个文件")
            return output_files
            
        except Exception as e:
            logger.error(f"处理目录时出错: {str(e)}", exc_info=True)
            raise


def main():
    """主函数，处理命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description='PDF知识点提炼工具')
    
    # 添加参数
    parser.add_argument('input', help='输入的PDF文件或包含PDF文件的目录')
    parser.add_argument('-o', '--output-dir', help='输出目录')
    parser.add_argument('-f', '--format', choices=['markdown', 'json', 'txt'], 
                        default='markdown', help='输出格式 (默认: markdown)')
    parser.add_argument('-m', '--mode', choices=['auto', 'keywords', 'sentences', 'sections'],
                        default='auto', help='知识点提取模式 (默认: auto)')
    parser.add_argument('-i', '--importance', type=float, default=0.5,
                        help='知识点重要性阈值，范围0.0-1.0 (默认: 0.5)')
    
    args = parser.parse_args()
    
    try:
        tool = PDFKnowledgeTool()
        
        if os.path.isdir(args.input):
            # 处理目录
            output_files = tool.process_directory(
                args.input,
                output_format=args.format,
                extract_mode=args.mode,
                min_importance=args.importance,
                output_dir=args.output_dir
            )
            print(f"处理完成，输出 {len(output_files)} 个文件:")
            for output_file in output_files:
                print(f" - {output_file}")
                
        elif os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
            # 处理单个文件
            output_file = tool.process_pdf(
                args.input,
                output_format=args.format,
                extract_mode=args.mode,
                min_importance=args.importance,
                output_dir=args.output_dir
            )
            print(f"处理完成，输出文件: {output_file}")
            
        else:
            print(f"错误: 输入必须是PDF文件或包含PDF文件的目录: {args.input}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {str(e)}", exc_info=True)
        print(f"错误: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())