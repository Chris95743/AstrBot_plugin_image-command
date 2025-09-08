from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.core.message.components import Reply
from .utils.ttp import generate_image_openrouter
from .utils.file_send_server import send_file
import asyncio, time
from collections import defaultdict, deque
from math import ceil

@register("gemini-25-image-openrouter", "喵喵", "使用openrouter的免费api生成图片", "1.6")
class MyPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        # 支持多个API密钥
        self.openrouter_api_keys = config.get("openrouter_api_keys", [])
        # 向后兼容：如果还在使用旧的单个API密钥配置
        old_api_key = config.get("openrouter_api_key")
        if old_api_key and not self.openrouter_api_keys:
            self.openrouter_api_keys = [old_api_key]
        
        # 自定义API base支持
        self.custom_api_base = config.get("custom_api_base", "").strip()
        
        # 模型配置
        self.model_name = config.get("model_name", "google/gemini-2.5-flash-image-preview:free").strip()
        
        # 重试配置
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        
        self.nap_server_address = config.get("nap_server_address")
        self.nap_server_port = config.get("nap_server_port")

        # 每群每分钟调用次数限制（<=0 表示不限制）
        self.calls_per_minute_per_group = int(config.get("calls_per_minute_per_group", 5) or 5)
        self._rate_buckets = defaultdict(deque)  # key -> deque[timestamps]
        self._rate_lock = asyncio.Lock()

    def _group_key(self, event: AstrMessageEvent) -> str:
        """根据事件生成分组键：优先群ID，其次会话ID/发送者ID。"""
        gid = None
        try:
            gid = event.get_group_id()
        except Exception:
            gid = None
        if gid:
            return f"group:{gid}"
        # 非群聊场景退化为会话/私聊维度
        sid = None
        try:
            sid = event.get_session_id()
        except Exception:
            sid = None
        if sid:
            return f"session:{sid}"
        try:
            sender = event.get_sender_id()
        except Exception:
            sender = "unknown"
        return f"private:{sender}"

    async def _try_acquire_rate(self, event: AstrMessageEvent):
        """尝试消耗一次配额。返回 (allowed: bool, wait_seconds: int, remaining: int)."""
        limit = self.calls_per_minute_per_group
        if limit <= 0:
            return True, 0, -1
        key = self._group_key(event)
        now = time.time()
        async with self._rate_lock:
            bucket = self._rate_buckets[key]
            # 清理 60 秒前的记录
            while bucket and (now - bucket[0]) >= 60:
                bucket.popleft()
            if len(bucket) >= limit:
                oldest = bucket[0]
                wait = ceil(max(0, 60 - (now - oldest)))
                return False, max(wait, 1), 0
            bucket.append(now)
            remaining = max(0, limit - len(bucket))
            return True, 0, remaining

    async def send_image_with_callback_api(self, image_path: str) -> Image:
        """
        优先使用callback_api_base发送图片，失败则退回到本地文件发送
        
        Args:
            image_path (str): 图片文件路径
            
        Returns:
            Image: 图片组件
        """
        callback_api_base = self.context.get_config().get("callback_api_base")
        if not callback_api_base:
            logger.info("未配置callback_api_base，使用本地文件发送")
            return Image.fromFileSystem(image_path)

        logger.info(f"检测到配置了callback_api_base: {callback_api_base}")
        try:
            image_component = Image.fromFileSystem(image_path)
            download_url = await image_component.convert_to_web_link()
            logger.info(f"成功生成下载链接: {download_url}")
            return Image.fromURL(download_url)
        except (IOError, OSError) as e:
            logger.warning(f"文件操作失败: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"网络连接失败: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)
        except Exception as e:
            logger.error(f"发送图片时出现未预期的错误: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)

    # 原函数工具已移除注册，避免与命令触发重复回复
    async def pic_gen(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True):
        """
            Generate or modify images using the Gemini model via the OpenRouter API.
            When a user requests image generation or drawing, call this function.
            If use_reference_images is True and the user has provided images in their message,
            those images will be used as references for generation or modification.
            If no images are provided or use_reference_images is False, pure text-to-image generation will be performed.

            Here are some examples:
            1. If the user wants to generate a large figure model, such as an anime character with normal proportions, please use a prompt like:
            "Please accurately transform the main subject in this photo into a realistic, masterpiece-like 1/7 scale PVC statue.
            A box should be placed beside the statue: the front of the box should have a large, clear transparent window printed with the main artwork, product name, brand logo, barcode, and a small specification or authenticity verification panel. A small price tag sticker must also be attached to the corner of the box. Meanwhile, a computer monitor should be placed at the back, and the monitor screen needs to display the ZBrush modeling process of this statue.
            In front of the packaging box, the statue should be placed on a round plastic base. The statue must have 3D dimensionality and a sense of realism, and the texture of the PVC material needs to be clearly represented. If the background can be set as an indoor scene, the effect will be even better.

            Below are detailed guidelines to note:
            When repairing any missing parts, there must be no poorly executed elements.
            When repairing human figures (if applicable), the body parts must be natural, movements must be coordinated, and the proportions of all parts must be reasonable.
            If the original photo is not a full-body shot, try to supplement the statue to make it a full-body version.
            The human figure's expression and movements must be exactly consistent with those in the photo.
            The figure's head should not appear too large, its legs should not appear too short, and the figure should not look stunted—this guideline may be ignored if the statue is a chibi-style design.
            For animal statues, the realism and level of detail of the fur should be reduced to make it more like a statue rather than the real original creature.
            No outer outline lines should be present, and the statue must not be flat.
            Please pay attention to the perspective relationship of near objects appearing larger and far objects smaller."

            2. If the user wants to generate a chibi figure model or a small, cute figure, please use a prompt like:
            "Please accurately transform the main subject in this photo into a realistic, masterpiece-like 1/7 scale PVC statue.
            Behind the side of this statue, a box should be placed: on the front of the box, the original image I entered, with the themed artwork, product name, brand logo, barcode, and a small specification or authenticity verification panel. A small price tag sticker must also be attached to one corner of the box. Meanwhile, a computer monitor should be placed at the back, and the monitor screen needs to display the ZBrush modeling process of this statue.
            In front of the packaging box, the statue should be placed on a round plastic base. The statue must have 3D dimensionality and a sense of realism, and the texture of the PVC material needs to be clearly represented. If the background can be set as an indoor scene, the effect will be even better.

            Below are detailed guidelines to note:
            When repairing any missing parts, there must be no poorly executed elements.
            When repairing human figures (if applicable), the body parts must be natural, movements must be coordinated, and the proportions of all parts must be reasonable.
            If the original photo is not a full-body shot, try to supplement the statue to make it a full-body version.
            The human figure's expression and movements must be exactly consistent with those in the photo.
            The figure's head should not appear too large, its legs should not appear too short, and the figure should not look stunted—this guideline may be ignored if the statue is a chibi-style design.
            For animal statues, the realism and level of detail of the fur should be reduced to make it more like a statue rather than the real original creature.
            No outer outline lines should be present, and the statue must not be flat.
            Please pay attention to the perspective relationship of near objects appearing larger and far objects smaller."

            Args:
            - image_description (string): Description of the image to generate. Translate to English can be better.
            - use_reference_images (bool): Whether to use images from the user's message as reference. Default True.
        """
        openrouter_api_keys = self.openrouter_api_keys
        nap_server_address = self.nap_server_address
        nap_server_port = self.nap_server_port

        # 根据参数决定是否使用参考图片
        input_images = []
        if use_reference_images:
            # 从当前对话上下文中获取图片信息
            if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
                for comp in event.message_obj.message:
                    if isinstance(comp, Image):
                        try:
                            base64_data = await comp.convert_to_base64()
                            input_images.append(base64_data)
                        except (IOError, ValueError, OSError) as e:
                            logger.warning(f"转换当前消息中的参考图片到base64失败: {e}")
                        except Exception as e:
                            logger.error(f"处理当前消息中的图片时出现未预期的错误: {e}")
                    elif isinstance(comp, Reply):
                        # 修复引用消息中的图片获取逻辑
                        # Reply组件的chain字段包含被引用的消息内容
                        if comp.chain:
                            for reply_comp in comp.chain:
                                if isinstance(reply_comp, Image):
                                    try:
                                        base64_data = await reply_comp.convert_to_base64()
                                        input_images.append(base64_data)
                                        logger.info(f"从引用消息中获取到图片")
                                    except (IOError, ValueError, OSError) as e:
                                        logger.warning(f"转换引用消息中的参考图片到base64失败: {e}")
                                    except Exception as e:
                                        logger.error(f"处理引用消息中的图片时出现未预期的错误: {e}")
                        else:
                            logger.debug("引用消息的chain为空，无法获取图片内容")
            
            # 记录使用的图片数量
            if input_images:
                logger.info(f"使用了 {len(input_images)} 张参考图片进行图像生成")
            else:
                logger.info("未找到参考图片，执行纯文本图像生成")

        # 调用生成图像的函数
        try:
            image_url, image_path = await generate_image_openrouter(
                image_description,
                openrouter_api_keys,
                model=self.model_name,
                input_images=input_images,
                api_base=self.custom_api_base if self.custom_api_base else None,
                max_retry_attempts=self.max_retry_attempts
            )
            
            if not image_url or not image_path:
                # 生成失败，发送错误消息
                error_chain = [Plain("图像生成失败，请检查API配置和网络连接。")]
                yield event.chain_result(error_chain)
                return
            
            # 处理文件传输和图片发送
            if self.nap_server_address and self.nap_server_address != "localhost":
                image_path = await send_file(image_path, HOST=nap_server_address, PORT=nap_server_port)
            
            # 使用新的发送方法，优先使用callback_api_base
            image_component = await self.send_image_with_callback_api(image_path)
            chain = [image_component]
            yield event.chain_result(chain)
            return
                
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"网络连接错误导致图像生成失败: {e}")
            error_chain = [Plain(f"网络连接错误，图像生成失败: {str(e)}")]
            yield event.chain_result(error_chain)
            return
        except ValueError as e:
            logger.error(f"参数错误导致图像生成失败: {e}")
            error_chain = [Plain(f"参数错误，图像生成失败: {str(e)}")]
            yield event.chain_result(error_chain)
            return
        except Exception as e:
            logger.error(f"图像生成过程出现未预期的错误: {e}")
            error_chain = [Plain(f"图像生成失败: {str(e)}")]
            yield event.chain_result(error_chain)
            return

    # 通过指令触发：/aiimg <描述>
    @filter.command("aiimg")
    async def aiimg(self, event: AstrMessageEvent, prompt: str = ""):
        """使用命令触发图片生成。示例：/aiimg 一只可爱的猫在草地上"""
        # 禁用默认 LLM 自动回复，避免与指令输出冲突
        try:
            event.should_call_llm(False)
        except Exception:
            pass

        allowed, wait, remaining = await self._try_acquire_rate(event)
        if not allowed:
            yield event.plain_result(
                f"频率限制：本群每分钟最多调用 {self.calls_per_minute_per_group} 次，请 {wait} 秒后再试。"
            )
            return

        desc = (prompt or "").strip()
        if not desc:
            # 兼容：从原始消息中截取指令后的文本
            try:
                raw = (event.message_str or "").strip()
                if raw.startswith("/aiimg"):
                    desc = raw[len("/aiimg"):].strip()
            except Exception:
                pass
        if not desc:
            yield event.plain_result("请在指令后提供描述，例如：/aiimg 一只在草地上奔跑的柯基")
            return

        # 复用已有逻辑，支持引用消息中的图片作为参考
        async for result in self.pic_gen(event, image_description=desc, use_reference_images=True):
            yield result

    # 通过指令触发：/aiimg手办化
    @filter.command("aiimg手办化")
    async def aiimg_shouban(self, event: AstrMessageEvent):
        """将参考图片“手办化”。用法：发送图片并输入 /aiimg手办化，或引用含图片的消息再输入指令。"""
        # 禁用默认 LLM 自动回复
        try:
            event.should_call_llm(False)
        except Exception:
            pass

        allowed, wait, remaining = await self._try_acquire_rate(event)
        if not allowed:
            yield event.plain_result(
                f"频率限制：本群每分钟最多调用 {self.calls_per_minute_per_group} 次，请 {wait} 秒后再试。"
            )
            return

        # 检查是否携带参考图片（当前消息或引用消息）
        has_image = False
        try:
            if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
                for comp in event.message_obj.message:
                    if isinstance(comp, Image):
                        has_image = True
                        break
                    if isinstance(comp, Reply) and getattr(comp, 'chain', None):
                        for reply_comp in comp.chain:
                            if isinstance(reply_comp, Image):
                                has_image = True
                                break
                    if has_image:
                        break
        except Exception:
            pass

        if not has_image:
            yield event.plain_result(
                "请先发送一张图片，或引用包含图片的消息后再使用指令：/aiimg手办化"
            )
            return

        # 固定提示词：用于以参考图为基础重制“手办化”效果
        prompt_text = (
            "将画面中的角色重塑为顶级收藏级树脂手办，全身动态姿势，置于角色主题底座；"
            "高精度材质，手工涂装，肌肤纹理与服装材质真实分明。"
            "戏剧性硬光为主光源，凸显立体感，无过曝；强效补光消除死黑，细节完整可见。"
            "背景为窗边景深模糊，侧后方隐约可见产品包装盒。"
            "博物馆级摄影质感，全身细节无损，面部结构精准。"
            "禁止：任何2D元素或照搬原图、塑料感、面部模糊、五官错位、细节丢失。"
        )

        async for result in self.pic_gen(event, image_description=prompt_text, use_reference_images=True):
            yield result
