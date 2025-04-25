"""
语音合成与动画制作模块
------------------
用于生成自然语音讲解和配套动画效果，
实现PPT内容的语音讲解和动态可视化展示。
"""

import os
import json
import re
import time
import base64
import tempfile
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

# 尝试导入必要的库
try:
    # 语音合成库
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False
    logging.warning("Azure TTS 库未安装，将使用替代方案")

try:
    # 音频处理库
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logging.warning("音频处理库未安装，部分音频处理功能将受限")

try:
    # 动画效果库
    from moviepy.editor import *
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy库未安装，动画生成功能将受限")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSProvider:
    """语音合成提供者基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def synthesize(self, text: str, output_path: str, voice_name: str = None) -> str:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本
            output_path: 输出的音频文件路径
            voice_name: 语音名称（可选）
            
        Returns:
            生成的音频文件路径
        """
        raise NotImplementedError("子类必须实现此方法")

class AzureTTSProvider(TTSProvider):
    """Azure语音服务提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 初始化Azure语音服务
        if not AZURE_TTS_AVAILABLE:
            raise ImportError("Azure TTS 库未安装")
            
        self.subscription_key = self.config.get('subscription_key')
        self.region = self.config.get('region')
        
        if not self.subscription_key or not self.region:
            raise ValueError("Azure TTS 需要subscription_key和region")
            
        # 初始化语音配置
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key, 
            region=self.region
        )
    
    def synthesize(self, text: str, output_path: str, voice_name: str = None) -> str:
        """使用Azure语音服务合成语音"""
        # 设置语音
        if voice_name:
            self.speech_config.speech_synthesis_voice_name = voice_name
        else:
            # 默认使用中文女声
            self.speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"
        
        # 设置输出配置
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        
        # 创建语音合成器
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )
        
        # 添加SSML标记提高表现力
        ssml_text = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
            <voice name="{self.speech_config.speech_synthesis_voice_name}">
                <prosody rate="-5%" pitch="+0%">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        # 合成语音
        result = synthesizer.speak_ssml_async(ssml_text).get()
        
        # 检查结果
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"语音合成完成: {output_path}")
            return output_path
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"语音合成取消: {cancellation_details.reason}")
            logger.error(f"错误详情: {cancellation_details.error_details}")
            raise Exception(f"语音合成失败: {cancellation_details.error_details}")
        else:
            logger.error(f"语音合成失败: {result.reason}")
            raise Exception(f"语音合成失败: {result.reason}")

class LocalTTSProvider(TTSProvider):
    """本地TTS备选方案（使用pyttsx3）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # 配置语音属性
            self.engine.setProperty('rate', 150)  # 语速
            self.engine.setProperty('volume', 0.9)  # 音量
            
            # 尝试设置中文语音
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        except ImportError:
            raise ImportError("pyttsx3库未安装，无法使用本地TTS")
    
    def synthesize(self, text: str, output_path: str, voice_name: str = None) -> str:
        """使用本地TTS引擎合成语音"""
        try:
            # 如果指定了语音名称，尝试设置
            if voice_name:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if voice_name.lower() in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # 合成语音并保存
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            
            logger.info(f"语音合成完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"本地TTS合成失败: {e}")
            raise

class AnimationGenerator:
    """动画生成器基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def generate_animation(self, content: Dict[str, Any], output_path: str) -> str:
        """
        生成动画效果
        
        Args:
            content: 动画内容数据
            output_path: 输出的动画文件路径
            
        Returns:
            生成的动画文件路径
        """
        raise NotImplementedError("子类必须实现此方法")

class MoviePyAnimator(AnimationGenerator):
    """使用MoviePy实现的动画生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy库未安装")
    
    def generate_animation(self, content: Dict[str, Any], output_path: str) -> str:
        """使用MoviePy生成动画"""
        animation_type = content.get('type', 'slide')
        
        if animation_type == 'slide':
            return self._generate_slide_animation(content, output_path)
        elif animation_type == 'concept':
            return self._generate_concept_animation(content, output_path)
        elif animation_type == 'graph':
            return self._generate_graph_animation(content, output_path)
        else:
            logger.warning(f"不支持的动画类型: {animation_type}")
            return self._generate_default_animation(content, output_path)
    
    def _generate_slide_animation(self, content: Dict[str, Any], output_path: str) -> str:
        """生成幻灯片过渡动画"""
        try:
            # 获取幻灯片图像
            slide_image = content.get('image_path')
            if not slide_image or not os.path.exists(slide_image):
                raise ValueError(f"幻灯片图像不存在: {slide_image}")
            
            # 创建视频剪辑
            slide_clip = ImageClip(slide_image).set_duration(content.get('duration', 5))
            
            # 应用淡入效果
            slide_clip = slide_clip.crossfadein(0.5)
            
            # 添加音频（如果有）
            audio_path = content.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                slide_clip = slide_clip.set_audio(audio_clip)
                
                # 调整视频时长以匹配音频
                slide_clip = slide_clip.set_duration(audio_clip.duration)
            
            # 导出视频
            slide_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac'
            )
            
            logger.info(f"幻灯片动画已生成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成幻灯片动画失败: {e}")
            raise
    
    def _generate_concept_animation(self, content: Dict[str, Any], output_path: str) -> str:
        """生成概念展示动画"""
        try:
            # 获取概念图像
            concept_image = content.get('image_path')
            if not concept_image or not os.path.exists(concept_image):
                raise ValueError(f"概念图像不存在: {concept_image}")
            
            # 创建图像剪辑
            concept_clip = ImageClip(concept_image).set_duration(content.get('duration', 10))
            
            # 获取概念文本
            concepts = content.get('concepts', [])
            text_clips = []
            
            # 创建文本动画
            for i, concept in enumerate(concepts):
                # 创建文本剪辑
                txt_clip = TextClip(
                    concept,
                    fontsize=30,
                    color='white',
                    font='Arial-Bold',
                    stroke_color='black',
                    stroke_width=1
                )
                
                # 设置位置和时长
                start_time = i * 2  # 每个概念间隔2秒
                txt_clip = (
                    txt_clip
                    .set_position(('center', 100 + i * 50))
                    .set_start(start_time)
                    .set_duration(2)
                    .crossfadein(0.5)
                    .crossfadeout(0.5)
                )
                
                text_clips.append(txt_clip)
            
            # 合成所有剪辑
            final_clip = CompositeVideoClip([concept_clip] + text_clips)
            
            # 添加音频（如果有）
            audio_path = content.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                final_clip = final_clip.set_audio(audio_clip)
                
                # 调整视频时长以匹配音频
                final_clip = final_clip.set_duration(max(final_clip.duration, audio_clip.duration))
            
            # 导出视频
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac'
            )
            
            logger.info(f"概念动画已生成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成概念动画失败: {e}")
            raise
def _generate_graph_animation(self, content: Dict[str, Any], output_path: str) -> str:
        """生成图形/图表动画"""
        try:
            # 获取图形图像
            graph_image = content.get('image_path')
            if not graph_image or not os.path.exists(graph_image):
                raise ValueError(f"图形图像不存在: {graph_image}")
            
            # 创建图像剪辑
            graph_clip = ImageClip(graph_image).set_duration(content.get('duration', 15))
            
            # 应用缩放动画效果
            def zoom_effect(t):
                zoom_factor = 1 + 0.1 * t  # 逐渐放大
                return zoom_factor
            
            graph_clip = graph_clip.resize(lambda t: zoom_effect(t))
            
            # 添加音频（如果有）
            audio_path = content.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                graph_clip = graph_clip.set_audio(audio_clip)
                
                # 调整视频时长以匹配音频
                graph_clip = graph_clip.set_duration(audio_clip.duration)
            
            # 导出视频
            graph_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac'
            )
            
            logger.info(f"图形动画已生成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成图形动画失败: {e}")
            raise
    
    def _generate_default_animation(self, content: Dict[str, Any], output_path: str) -> str:
        """生成默认动画"""
        try:
            # 创建一个彩色背景
            color_clip = ColorClip(
                size=(1280, 720),
                color=[0, 0, 255],  # 蓝色背景
                duration=content.get('duration', 5)
            )
            
            # 添加标题文本
            title = content.get('title', '教材内容')
            txt_clip = TextClip(
                title,
                fontsize=60,
                color='white',
                font='Arial-Bold'
            )
            
            # 设置文本位置和动画效果
            txt_clip = (
                txt_clip
                .set_position('center')
                .set_duration(color_clip.duration)
                .crossfadein(0.5)
                .crossfadeout(0.5)
            )
            
            # 合成剪辑
            final_clip = CompositeVideoClip([color_clip, txt_clip])
            
            # 添加音频（如果有）
            audio_path = content.get('audio_path')
            if audio_path and os.path.exists(audio_path):
                audio_clip = AudioFileClip(audio_path)
                final_clip = final_clip.set_audio(audio_clip)
                
                # 调整视频时长以匹配音频
                final_clip = final_clip.set_duration(audio_clip.duration)
            
            # 导出视频
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac'
            )
            
            logger.info(f"默认动画已生成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"生成默认动画失败: {e}")
            raise

class HTMLAnimationGenerator(AnimationGenerator):
    """生成HTML+CSS+JS动画（作为备选方案）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
    
    def generate_animation(self, content: Dict[str, Any], output_path: str) -> str:
        """生成HTML动画"""
        animation_type = content.get('type', 'slide')
        
        if animation_type == 'slide':
            return self._generate_slide_html(content, output_path)
        elif animation_type == 'concept':
            return self._generate_concept_html(content, output_path)
        elif animation_type == 'graph':
            return self._generate_graph_html(content, output_path)
        else:
            return self._generate_default_html(content, output_path)
    
    def _generate_slide_html(self, content: Dict[str, Any], output_path: str) -> str:
        """生成幻灯片HTML动画"""
        # 获取图像路径
        image_path = content.get('image_path', '')
        image_data = ''
        
        if image_path and os.path.exists(image_path):
            # 读取图像并转换为Base64
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # 获取音频路径
        audio_path = content.get('audio_path', '')
        audio_tag = ''
        
        if audio_path and os.path.exists(audio_path):
            audio_tag = f'<audio id="audio" autoplay controls><source src="{audio_path}" type="audio/mpeg"></audio>'
        
        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{content.get('title', '幻灯片')}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }}
                .slide-container {{
                    width: 80%;
                    max-width: 1024px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                    animation: fadeIn 1s ease-in;
                }}
                .slide {{
                    width: 100%;
                    background-color: white;
                    padding: 20px;
                    box-sizing: border-box;
                }}
                .slide img {{
                    width: 100%;
                    height: auto;
                    opacity: 0;
                    animation: fadeIn 1s ease-in forwards;
                }}
                .audio-container {{
                    margin-top: 20px;
                    text-align: center;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
            </style>
        </head>
        <body>
            <div class="slide-container">
                <div class="slide">
                    <img src="data:image/png;base64,{image_data}" alt="{content.get('title', '幻灯片')}">
                </div>
                <div class="audio-container">
                    {audio_tag}
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML幻灯片已生成: {output_path}")
        return output_path
    
    def _generate_concept_html(self, content: Dict[str, Any], output_path: str) -> str:
        """生成概念展示HTML动画"""
        # 获取图像路径
        image_path = content.get('image_path', '')
        image_data = ''
        
        if image_path and os.path.exists(image_path):
            # 读取图像并转换为Base64
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # 获取音频路径
        audio_path = content.get('audio_path', '')
        audio_tag = ''
        
        if audio_path and os.path.exists(audio_path):
            audio_tag = f'<audio id="audio" autoplay controls><source src="{audio_path}" type="audio/mpeg"></audio>'
        
        # 获取概念列表
        concepts = content.get('concepts', [])
        concepts_html = ''
        
        for i, concept in enumerate(concepts):
            delay = i * 0.5  # 每个概念的延迟
            concepts_html += f"""
            <div class="concept" style="animation-delay: {delay}s;">
                <div class="concept-text">{concept}</div>
            </div>
            """
        
        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{content.get('title', '概念展示')}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }}
                .container {{
                    width: 80%;
                    max-width: 1024px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                    position: relative;
                }}
                .image-container {{
                    width: 100%;
                }}
                .image-container img {{
                    width: 100%;
                    height: auto;
                }}
                .concepts-container {{
                    position: absolute;
                    top: 50px;
                    left: 0;
                    width: 100%;
                    padding: 20px;
                    box-sizing: border-box;
                }}
                .concept {{
                    opacity: 0;
                    animation: fadeInUp 1s ease forwards;
                    margin-bottom: 20px;
                }}
                .concept-text {{
                    background-color: rgba(255,255,255,0.8);
                    padding: 10px 15px;
                    border-radius: 5px;
                    display: inline-block;
                    font-weight: bold;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }}
                .audio-container {{
                    margin-top: 20px;
                    text-align: center;
                }}
                @keyframes fadeInUp {{
                    from {{
                        opacity: 0;
                        transform: translateY(20px);
                    }}
                    to {{
                        opacity: 1;
                        transform: translateY(0);
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="image-container">
                    <img src="data:image/png;base64,{image_data}" alt="{content.get('title', '概念图')}">
                </div>
                <div class="concepts-container">
                    {concepts_html}
                </div>
                <div class="audio-container">
                    {audio_tag}
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML概念展示已生成: {output_path}")
        return output_path
    
    def _generate_graph_html(self, content: Dict[str, Any], output_path: str) -> str:
        """生成图表HTML动画"""
        # 获取图像路径
        image_path = content.get('image_path', '')
        image_data = ''
        
        if image_path and os.path.exists(image_path):
            # 读取图像并转换为Base64
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # 获取音频路径
        audio_path = content.get('audio_path', '')
        audio_tag = ''
        
        if audio_path and os.path.exists(audio_path):
            audio_tag = f'<audio id="audio" autoplay controls><source src="{audio_path}" type="audio/mpeg"></audio>'
        
        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{content.get('title', '知识图谱')}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }}
                .container {{
                    width: 90%;
                    max-width: 1200px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                    background-color: white;
                    padding: 20px;
                    box-sizing: border-box;
                }}
                .title {{
                    text-align: center;
                    margin-bottom: 20px;
                    color: #333;
                    font-size: 24px;
                }}
                .graph-container {{
                    width: 100%;
                    overflow: hidden;
                }}
                .graph-container img {{
                    width: 100%;
                    height: auto;
                    transform: scale(0.8);
                    animation: zoomIn 3s ease forwards;
                }}
                .audio-container {{
                    margin-top: 20px;
                    text-align: center;
                }}
                @keyframes zoomIn {{
                    from {{
                        transform: scale(0.8);
                    }}
                    to {{
                        transform: scale(1);
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="title">{content.get('title', '知识图谱')}</div>
                <div class="graph-container">
                    <img src="data:image/png;base64,{image_data}" alt="{content.get('title', '知识图谱')}">
                </div>
                <div class="audio-container">
                    {audio_tag}
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML图表已生成: {output_path}")
        return output_path
    
    def _generate_default_html(self, content: Dict[str, Any], output_path: str) -> str:
        """生成默认HTML动画"""
        # 获取音频路径
        audio_path = content.get('audio_path', '')
        audio_tag = ''
        
        if audio_path and os.path.exists(audio_path):
            audio_tag = f'<audio id="audio" autoplay controls><source src="{audio_path}" type="audio/mpeg"></audio>'
        
        # 创建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{content.get('title', '教材内容')}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                    background-color: #0066cc;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    color: white;
                }}
                .container {{
                    text-align: center;
                }}
                .title {{
                    font-size: 48px;
                    margin-bottom: 30px;
                    opacity: 0;
                    animation: fadeIn 1s ease forwards;
                }}
                .audio-container {{
                    margin-top: 40px;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="title">{content.get('title', '教材内容')}</div>
                <div class="audio-container">
                    {audio_tag}
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"默认HTML动画已生成: {output_path}")
        return output_path

class MultimediaGenerator:
    """整合语音合成和动画生成的多媒体生成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 初始化TTS提供商
        self.tts_provider = self._init_tts_provider()
        
        # 初始化动画生成器
        self.animator = self._init_animator()
        
        # 临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 语音设置
        self.voice_name = self.config.get('voice_name')
    
    def _init_tts_provider(self) -> TTSProvider:
        """初始化语音合成提供者"""
        tts_type = self.config.get('tts_type', 'local')
        
        if tts_type == 'azure' and AZURE_TTS_AVAILABLE:
            try:
                # 获取Azure TTS配置
                azure_config = {
                    'subscription_key': self.config.get('azure_subscription_key'),
                    'region': self.config.get('azure_region')
                }
                
                return AzureTTSProvider(azure_config)
            except Exception as e:
                logger.warning(f"Azure TTS初始化失败: {e}，将使用本地TTS")
        
        # 默认使用本地TTS
        try:
            return LocalTTSProvider(self.config)
        except Exception as e:
            logger.error(f"无法初始化TTS提供者: {e}")
            raise
    
    def _init_animator(self) -> AnimationGenerator:
        """初始化动画生成器"""
        animation_type = self.config.get('animation_type', 'html')
        
        if animation_type == 'moviepy' and MOVIEPY_AVAILABLE:
            try:
                return MoviePyAnimator(self.config)
            except Exception as e:
                logger.warning(f"MoviePy动画生成器初始化失败: {e}，将使用HTML动画生成器")
        
        # 默认使用HTML动画生成器
        return HTMLAnimationGenerator(self.config)
    
    def generate_multimedia(self, content_analysis: Dict[str, Any], 
                           ppt_image_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        为PPT内容生成语音和动画
        
        Args:
            content_analysis: 内容分析结果
            ppt_image_dir: PPT图像目录（包含幻灯片截图）
            output_dir: 输出目录
            
        Returns:
            多媒体生成结果
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建音频和动画目录
        audio_dir = os.path.join(output_dir, 'audio')
        animation_dir = os.path.join(output_dir, 'animation')
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(animation_dir, exist_ok=True)
        
        # 初始化结果
        result = {
            'slides': [],
            'audio_files': [],
            'animation_files': []
        }
        
        # 获取章节
        chapters = content_analysis.get('chapters', [])
        
        # 生成封面语音和动画
        cover_result = self._generate_cover_multimedia(
            content_analysis, 
            ppt_image_dir, 
            audio_dir, 
            animation_dir
        )
        result['slides'].append(cover_result)
        
        # 为每个章节生成语音和动画
        for chapter in chapters:
            chapter_result = self._generate_chapter_multimedia(
                chapter, 
                content_analysis.get('subject', 'general'),
                ppt_image_dir, 
                audio_dir, 
                animation_dir
            )
            result['slides'].extend(chapter_result)
        
        # 为知识图谱生成语音和动画
        if 'knowledge_graph' in content_analysis:
            graph_result = self._generate_graph_multimedia(
                content_analysis,
                ppt_image_dir,
                audio_dir,
                animation_dir
            )
            result['slides'].append(graph_result)
        
        # 收集所有音频和动画文件
        for slide in result['slides']:
            if 'audio_path' in slide and slide['audio_path']:
                result['audio_files'].append(slide['audio_path'])
            if 'animation_path' in slide and slide['animation_path']:
                result['animation_files'].append(slide['animation_path'])
        
        return result
    
    def _generate_cover_multimedia(self, content_analysis: Dict[str, Any], 
                                  ppt_dir: str, audio_dir: str, 
                                  animation_dir: str) -> Dict[str, Any]:
        """为封面幻灯片生成语音和动画"""
        # 找到封面图像
        cover_image = os.path.join(ppt_dir, 'slide_1.png')
        if not os.path.exists(cover_image):
            cover_image = self._find_slide_image(ppt_dir, 1)
            if not cover_image:
                logger.warning("找不到封面图像")
                return {}
        
        # 生成封面介绍文本
        subject = content_analysis.get('subject', 'general')
        if subject == 'physics':
            subject_text = "物理"
        elif subject == 'chemistry':
            subject_text = "化学"
        elif subject == 'biology':
            subject_text = "生物"
        elif subject == 'math':
            subject_text = "数学"
        else:
            subject_text = subject.capitalize()
        
        main_topics = content_analysis.get('main_topics', [])
        title = "教材内容讲解"
        if main_topics and len(main_topics) > 0:
            title = main_topics[0]
        
        cover_text = f"欢迎来到{subject_text}课程：{title}。这堂课我们将学习这本教材的主要内容，通过互动的方式帮助你掌握重要知识点。"
        
        # 生成语音
        audio_path = os.path.join(audio_dir, 'cover.mp3')
        try:
            self.tts_provider.synthesize(cover_text, audio_path, self.voice_name)
        except Exception as e:
            logger.error(f"封面语音生成失败: {e}")
            audio_path = None
        
        # 生成动画
        animation_path = os.path.join(animation_dir, 'cover.html')
        try:
            animation_content = {
                'type': 'slide',
                'title': title,
                'image_path': cover_image,
                'audio_path': audio_path,
                'duration': 10
            }
            self.animator.generate_animation(animation_content, animation_path)
        except Exception as e:
            logger.error(f"封面动画生成失败: {e}")
            animation_path = None
        
        return {
            'title': title,
            'type': 'cover',
            'image_path': cover_image,
            'audio_path': audio_path,
            'animation_path': animation_path,
            'text': cover_text
        }
    
    def _generate_chapter_multimedia(self, chapter: Dict[str, Any], subject: str,
                                    ppt_dir: str, audio_dir: str, 
                                    animation_dir: str) -> List[Dict[str, Any]]:
        """为章节生成语音和动画"""
        result = []
        
        # 章节标题
        chapter_title = chapter.get('title', '未命名章节')
        
        # 生成章节标题幻灯片的语音和动画
        title_slide = self._generate_title_slide_multimedia(
            chapter, 
            subject,
            ppt_dir, 
            audio_dir, 
            animation_dir
        )
        if title_slide:
            result.append(title_slide)
        
        # 生成关键概念幻灯片的语音和动画
        key_concepts = chapter.get('key_concepts', [])
        if key_concepts:
            concept_slide = self._generate_concept_slide_multimedia(
                chapter,
                key_concepts,
                ppt_dir,
                audio_dir,
                animation_dir
            )
            if concept_slide:
                result.append(concept_slide)
        
        # 生成学习目标幻灯片的语音和动画
        learning_objectives = chapter.get('learning_objectives', [])
        if learning_objectives:
            objective_slide = self._generate_objective_slide_multimedia(
                chapter,
                learning_objectives,
                ppt_dir,
                audio_dir,
                animation_dir
            )
            if objective_slide:
                result.append(objective_slide)
        
        # 如果有子章节，递归生成
        subchapters = chapter.get('subchapters', [])
        for subchapter in subchapters:
            sub_results = self._generate_chapter_multimedia(
                subchapter,
                subject,
                ppt_dir,
                audio_dir,
                animation_dir
            )
            result.extend(sub_results)
        
        return result
    
    def _generate_title_slide_multimedia(self, chapter: Dict[str, Any], subject: str,
                                        ppt_dir: str, audio_dir: str, 
                                        animation_dir: str) -> Dict[str, Any]:
        """为章节标题幻灯片生成语音和动画"""
        # 章节标题
        chapter_title = chapter.get('title', '未命名章节')
        
        # 查找章节幻灯片图像
        slide_index = self._get_chapter_slide_index(chapter_title, ppt_dir)
        if slide_index == -1:
            logger.warning(f"找不到章节 '{chapter_title}' 的幻灯片图像")
            return {}
        
        slide_image = os.path.join(ppt_dir, f'slide_{slide_index}.png')
        if not os.path.exists(slide_image):
            slide_image = self._find_slide_image(ppt_dir, slide_index)
            if not slide_image:
                logger.warning(f"找不到章节 '{chapter_title}' 的幻灯片图像")
                return {}
        
        # 生成章节介绍文本
        chapter_summary = chapter.get('summary', '')
        difficulty = chapter.get('difficulty_level', 'intermediate')
        
        if difficulty == 'beginner':
            difficulty_text = "入门级别"
        elif difficulty == 'advanced':
            difficulty_text = "高级内容"
        else:
            difficulty_text = "中级内容"
        
        title_text = f"我们现在开始学习{chapter_title}。这部分是{difficulty_text}。"
        
        if chapter_summary:
            title_text += f" {chapter_summary}"
        
        # 生成语音
        audio_path = os.path.join(audio_dir, f'chapter_{slide_index}.mp3')
        try:
            self.tts_provider.synthesize(title_text, audio_path, self.voice_name)
        except Exception as e:
            logger.error(f"章节标题语音生成失败: {e}")
            audio_path = None
        
        # 生成动画
        animation_path = os.path.join(animation_dir, f'chapter_{slide_index}.html')
        try:
            animation_content = {
                'type': 'slide',
                'title': chapter_title,
                'image_path': slide_image,
                'audio_path': audio_path,
                'duration': 8
            }
            self.animator.generate_animation(animation_content, animation_path)
        except Exception as e:
            logger.error(f"章节标题动画生成失败: {e}")
            animation_path = None
        
        return {
            'title': chapter_title,
            'type': 'chapter',
            'slide_index': slide_index,
            'image_path': slide_image,
            'audio_path': audio_path,
            'animation_path': animation_path,
            'text': title_text
        }
    
    def _generate_concept_slide_multimedia(self, chapter: Dict[str, Any], 
                                          concepts: List[str],
                                          ppt_dir: str, audio_dir: str, 
                                          animation_dir: str) -> Dict[str, Any]:
        """为关键概念幻灯片生成语音和动画"""
        # 章节标题
        chapter_title = chapter.get('title', '未命名章节')
        
        # 查找概念幻灯片图像
        slide_index = self._get_concept_slide_index(chapter_title, ppt_dir)
        if slide_index == -1:
            logger.warning(f"找不到章节 '{chapter_title}' 的概念幻灯片图像")
            return {}
        
        slide_image = os.path.join(ppt_dir, f'slide_{slide_index}.png')
        if not os.path.exists(slide_image):
            slide_image = self._find_slide_image(ppt_dir, slide_index)
            if not slide_image:
                logger.warning(f"找不到章节 '{chapter_title}' 的概念幻灯片图像")
                return {}
        
        # 生成概念介绍文本
        concept_text = f"在{chapter_title}中，我们需要掌握以下关键概念："
        
        for i, concept in enumerate(concepts[:5]):  # 限制为前5个概念
            concept_text += f" {i+1}，{concept}。"
        
        concept_text += " 让我们一起来理解这些重要的知识点。"
        
        # 生成语音
        audio_path = os.path.join(audio_dir, f'concepts_{slide_index}.mp3')
        try:
            self.tts_provider.synthesize(concept_text, audio_path, self.voice_name)
        except Exception as e:
            logger.error(f"概念语音生成失败: {e}")
            audio_path = None
        
        # 生成动画
        animation_path = os.path.join(animation_dir, f'concepts_{slide_index}.html')
        try:
            animation_content = {
                'type': 'concept',
                'title': f"{chapter_title} - 关键概念",
                'image_path': slide_image,
                'audio_path': audio_path,
                'concepts': concepts[:5],
                'duration': 10
            }
            self.animator.generate_animation(animation_content, animation_path)
        except Exception as e:
            logger.error(f"概念动画生成失败: {e}")
            animation_path = None
        
        return {
            'title': f"{chapter_title} - 关键概念",
            'type': 'concepts',
            'slide_index': slide_index,
            'image_path': slide_image,
            'audio_path': audio_path,
            'animation_path': animation_path,
            'text': concept_text,
            'concepts': concepts[:5]
        }
    
    def _generate_objective_slide_multimedia(self, chapter: Dict[str, Any], 
                                           objectives: List[str],
                                           ppt_dir: str, audio_dir: str, 
                                           animation_dir: str) -> Dict[str, Any]:
        """为学习目标幻灯片生成语音和动画"""
        # 章节标题
        chapter_title = chapter.get('title', '未命名章节')
        
        # 查找学习目标幻灯片图像
        slide_index = self._get_objective_slide_index(chapter_title, ppt_dir)
        if slide_index == -1:
            logger.warning(f"找不到章节 '{chapter_title}' 的学习目标幻灯片图像")
            return {}
        
        slide_image = os.path.join(ppt_dir, f'slide_{slide_index}.png')
        if not os.path.exists(slide_image):
            slide_image = self._find_slide_image(ppt_dir, slide_index)
            if not slide_image:
                logger.warning(f"找不到章节 '{chapter_title}' 的学习目标幻灯片图像")
                return {}
        
        # 生成学习目标介绍文本
        objective_text = f"学习{chapter_title}后，你将能够："
        
        for i, objective in enumerate(objectives):
            objective_text += f" {i+1}，{objective}。"
        
        # 生成语音
        audio_path = os.path.join(audio_dir, f'objectives_{slide_index}.mp3')
        try:
            self.tts_provider.synthesize(objective_text, audio_path, self.voice_name)
        except Exception as e:
            logger.error(f"学习目标语音生成失败: {e}")
            audio_path = None
        
        # 生成动画
        animation_path = os.path.join(animation_dir, f'objectives_{slide_index}.html')
        try:
            animation_content = {
                'type': 'slide',
                'title': f"{chapter_title} - 学习目标",
                'image_path': slide_image,
                'audio_path': audio_path,
                'duration': 8
            }
            self.animator.generate_animation(animation_content, animation_path)
        except Exception as e:
            logger.error(f"学习目标动画生成失败: {e}")
            animation_path = None
        
        return {
            'title': f"{chapter_title} - 学习目标",
            'type': 'objectives',
            'slide_index': slide_index,
            'image_path': slide_image,
            'audio_path': audio_path,
            'animation_path': animation_path,
            'text': objective_text,
            'objectives': objectives
        }
    
    def _generate_graph_multimedia(self, content_analysis: Dict[str, Any],
                                  ppt_dir: str, audio_dir: str, 
                                  animation_dir: str) -> Dict[str, Any]:
        """为知识图谱幻灯片生成语音和动画"""
        # 查找知识图谱幻灯片图像
        slide_index = self._get_graph_slide_index(ppt_dir)
        if slide_index == -1:
            logger.warning("找不到知识图谱幻灯片图像")
            return {}
        
        slide_image = os.path.join(ppt_dir, f'slide_{slide_index}.png')
        if not os.path.exists(slide_image):
            slide_image = self._find_slide_image(ppt_dir, slide_index)
            if not slide_image:
                logger.warning("找不到知识图谱幻灯片图像")
                return {}
        
        # 生成知识图谱介绍文本
        important_concepts = content_analysis.get('important_concepts', [])
        concepts_str = ""
        
        if important_concepts:
            concepts_str = "，".join([concept.get('concept', '') for concept in important_concepts[:3]])
            
        graph_text = f"这是整个教材的知识结构图。你可以看到各个概念之间的关联。特别重要的概念包括{concepts_str}等。理解这些概念之间的关系，有助于我们系统地掌握整个知识体系。"
        
        # 生成语音
        audio_path = os.path.join(audio_dir, f'graph_{slide_index}.mp3')
        try:
            self.tts_provider.synthesize(graph_text, audio_path, self.voice_name)
        except Exception as e:
            logger.error(f"知识图谱语音生成失败: {e}")
            audio_path = None
        
        # 生成动画
        animation_path = os.path.join(animation_dir, f'graph_{slide_index}.html')
        try:
            animation_content = {
                'type': 'graph',
                'title': "知识图谱",
                'image_path': slide_image,
                'audio_path': audio_path,
                'duration': 12
            }
            self.animator.generate_animation(animation_content, animation_path)
        except Exception as e:
            logger.error(f"知识图谱动画生成失败: {e}")
            animation_path = None
        
        return {
            'title': "知识图谱",
            'type': 'graph',
            'slide_index': slide_index,
            'image_path': slide_image,
            'audio_path': audio_path,
            'animation_path': animation_path,
            'text': graph_text
        }
    
    def _get_chapter_slide_index(self, chapter_title: str, ppt_dir: str) -> int:
        """获取章节标题幻灯片的索引"""
        # 在实际实现中，需要通过OCR或其他方式识别幻灯片标题
        # 这里简化实现，假设幻灯片按顺序排列
        
        # 模拟搜索逻辑
        for i in range(1, 100):  # 假设最多100张幻灯片
            slide_path = os.path.join(ppt_dir, f'slide_{i}.png')
            if os.path.exists(slide_path):
                # 这里应该使用OCR读取幻灯片内容
                # 简化实现，假设找到了匹配的幻灯片
                return i
        
        return -1
    
    def _get_concept_slide_index(self, chapter_title: str, ppt_dir: str) -> int:
        """获取关键概念幻灯片的索引"""
        # 简化实现
        chapter_index = self._get_chapter_slide_index(chapter_title, ppt_dir)
        if chapter_index > 0:
            return chapter_index + 1  # 假设关键概念幻灯片紧跟章节标题幻灯片
        
        return -1
    
    def _get_objective_slide_index(self, chapter_title: str, ppt_dir: str) -> int:
        """获取学习目标幻灯片的索引"""
        # 简化实现
        concept_index = self._get_concept_slide_index(chapter_title, ppt_dir)
        if concept_index > 0:
            return concept_index + 1  # 假设学习目标幻灯片紧跟关键概念幻灯片
        
        return -1
    
    def _get_graph_slide_index(self, ppt_dir: str) -> int:
        """获取知识图谱幻灯片的索引"""
        # 简化实现，假设知识图谱幻灯片是最后一张
        max_index = 0
        
        for i in range(1, 100):  # 假设最多100张幻灯片
            slide_path = os.path.join(ppt_dir, f'slide_{i}.png')
            if os.path.exists(slide_path):
                max_index = i
        
        if max_index > 0:
            return max_index
        
        return -1
    
    def _find_slide_image(self, ppt_dir: str, index: int) -> Optional[str]:
        """查找幻灯片图像"""
        # 尝试不同的文件名格式
        patterns = [
            f'slide_{index}.png',
            f'slide_{index}.jpg',
            f'slide{index}.png',
            f'slide{index}.jpg',
            f'Slide_{index}.png',
            f'Slide_{index}.jpg',
            f'幻灯片{index}.png',
            f'幻灯片{index}.jpg'
        ]
        
        for pattern in patterns:
            path = os.path.join(ppt_dir, pattern)
            if os.path.exists(path):
                return path
        
        return None
    
    def __del__(self):
        """清理临时目录"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成PPT内容的语音讲解和动画')
    parser.add_argument('content_file', help='内容分析结果JSON文件')
    parser.add_argument('ppt_image_dir', help='PPT幻灯片图像目录')
    parser.add_argument('--output_dir', '-o', help='输出目录')
    parser.add_argument('--tts_type', choices=['azure', 'local'], default='local', help='TTS类型')
    parser.add_argument('--azure_key', help='Azure语音服务密钥')
    parser.add_argument('--azure_region', help='Azure语音服务区域')
    parser.add_argument('--animation_type', choices=['moviepy', 'html'], default='html', help='动画生成类型')
    parser.add_argument('--voice_name', help='语音名称')
    
    args = parser.parse_args()
    
    # 加载内容分析结果
    with open(args.content_file, 'r', encoding='utf-8') as f:
        content_analysis = json.load(f)
    
    # 配置
    config = {
        'tts_type': args.tts_type,
        'animation_type': args.animation_type
    }
    
    if args.tts_type == 'azure':
        if not args.azure_key or not args.azure_region:
            parser.error("使用Azure TTS需要提供azure_key和azure_region")
        
        config['azure_subscription_key'] = args.azure_key
        config['azure_region'] = args.azure_region
    
    if args.voice_name:
        config['voice_name'] = args.voice_name
    
    # 设置默认输出目录
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(args.content_file), 'multimedia')
    
    # 初始化多媒体生成器
    multimedia_generator = MultimediaGenerator(config)
    
    # 生成多媒体内容
    result = multimedia_generator.generate_multimedia(
        content_analysis, 
        args.ppt_image_dir, 
        output_dir
    )
    
    # 保存结果元数据
    metadata_path = os.path.join(output_dir, 'multimedia_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"多媒体内容已生成: {output_dir}")
    print(f"元数据文件: {metadata_path}")
    print(f"共生成 {len(result['audio_files'])} 个音频文件和 {len(result['animation_files'])} 个动画文件")

if __name__ == "__main__":
    main()