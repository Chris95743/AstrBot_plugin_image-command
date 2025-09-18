from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.core.message.components import Reply, Image, Plain
import base64
import aiofiles
from .utils.ttp import generate_image_openrouter, generate_image_ark
from .utils.file_send_server import send_file
import asyncio, time
from collections import defaultdict, deque
from math import ceil
from typing import List, Dict, Any, Optional, Tuple, Deque, Union, AsyncGenerator

# 常量定义
class Constants:
    # 默认配置值
    DEFAULT_MODEL_NAME = "google/gemini-2.5-flash-image-preview:free"
    DEFAULT_MAX_RETRY_ATTEMPTS = 3
    DEFAULT_CALLS_PER_MINUTE_PER_GROUP = 5
    DEFAULT_PROVIDER = "openrouter"  # openrouter | ark
    
    # 速率限制相关
    RATE_LIMIT_WINDOW_SECONDS = 60
    MIN_WAIT_SECONDS = 1
    
    # 错误消息
    ERROR_MSG_GENERATION_FAILED = "图像生成失败，请检查API配置和网络连接。"
    ERROR_MSG_NETWORK_ERROR = "网络连接错误，图像生成失败: {}"
    ERROR_MSG_PARAM_ERROR = "参数错误，图像生成失败: {}"
    ERROR_MSG_UNKNOWN_ERROR = "图像生成失败: {}"
    ERROR_MSG_RATE_LIMIT = "频率限制：本群每分钟最多调用 {} 次，请 {} 秒后再试。"
    ERROR_MSG_NO_DESCRIPTION = "请在指令后提供描述，例如：/aiimg 一只在草地上奔跑的柯基"
    ERROR_MSG_NO_IMAGE = "请先发送一张图片，或引用包含图片的消息后再使用指令：/aiimg手办化"
    ERROR_MSG_CONFIG_ERROR = "配置加载错误: {}"
    ERROR_MSG_GROUP_NOT_ALLOWED = "此QQ群无权使用本插件。"
    ERROR_MSG_GROUP_BLACKLISTED = "此QQ群已被禁止使用本插件。"
    ERROR_MSG_ARK_NO_KEY = "未配置 Ark API 密钥。"
    ERROR_MSG_ARK_NO_KEY = "未配置 Ark API 密钥。"
    
    # 手办化固定提示词
    SHOUBAN_PROMPT = (
        "将这张图片改成手办，调整到符合背景的大小和位置，细节丰富，真实展现，背景有手办包装盒与电脑桌面场景，后方电脑显示 3D 建模软件界面，呈现对应手办的设计稿，突出手办的造型、色彩，还原桌面摆放氛围 ，使用透明亚克力的圆型底座"
        "禁止：任何2D元素或照搬原图、塑料感、面部模糊、五官错位、细节丢失。"
    )
    
    # 手办化2固定提示词（英文版本）
    SHOUBAN2_PROMPT = (
        "Create a highly realistic 1/7 scale commercialized figure based on the illustration's adult character, "
        "ensuring the appearance and content are safe, healthy, and free from any inappropriate elements. "
        "Render the figure in a detailed, lifelike style and environment, placed on a shelf inside an "
        "ultra-realistic figure display cabinet, mounted on a circular transparent acrylic base without any text. "
        "Maintain highly precise details in texture, material, and paintwork to enhance realism. "
        "The cabinet scene should feature a natural depth of field with a smooth transition between foreground "
        "and background for a realistic photographic look. Lighting should appear natural and adaptive to the scene, "
        "automatically adjusting based on the overall composition instead of being locked to a specific direction, "
        "simulating the quality and reflection of real commercial photography. Other shelves in the cabinet should "
        "contain different figures which are slightly blurred due to being out of focus, enhancing spatial realism and depth."
    )
    
    # 帮助信息
    HELP_MESSAGE = (
        "📖 Gemini图像生成插件帮助\n\n"
        "🎨 可用指令：\n"
        "1. /aiimg <描述> - 根据文字描述生成图像，支持参考图片\n"
        "   示例：/aiimg 画一只可爱的猫咪\n\n"
        "2. /aiimg手办化 - 将参考图片转换为手办风格(模版1)\n\n"
        "3. /aiimg手办化2 - 将参考图片转换为手办风格(模版2)\n\n"
        "4. /aiimg帮助 - 显示此帮助信息\n\n"
        "💡 提示：\n"
        "• 支持同时使用图片和文字描述进行生成\n"
        "• 可以引用其他消息中的图片作为参考\n"
        "• 手办化功能需要提供参考图片"
    )

class PluginError(Exception):
    """插件自定义异常基类"""
    pass

class ConfigError(PluginError):
    """配置相关错误"""
    pass

@register("image-command", "喵喵/薄暝/Chris95743", "修改自薄暝修改自喵喵的openrouter生图插件。使用openai格式的免费api生成图片(使用astrbot命令调用插件)", "v1.8")
class MyPlugin(Star):
    def __init__(self, context: Context, config: Dict[str, Any]) -> None:
        super().__init__(context)
        logger.info("图像生成插件初始化开始")
        
        try:
            # 支持多个API密钥
            self.openrouter_api_keys: List[str] = config.get("openrouter_api_keys", [])
            # 向后兼容：如果还在使用旧的单个API密钥配置
            old_api_key: Optional[str] = config.get("openrouter_api_key")
            if old_api_key and not self.openrouter_api_keys:
                self.openrouter_api_keys = [old_api_key]
            
            # 验证API密钥配置
            if not self.openrouter_api_keys:
                logger.warning("未配置任何API密钥，插件可能无法正常工作")
            
            # 自定义API base支持
            self.custom_api_base: str = config.get("custom_api_base", "").strip()
            
            # 提供方与 Ark 相关配置
            self.provider: str = (config.get("provider", Constants.DEFAULT_PROVIDER) or "openrouter").strip().lower()
            self.ark_api_keys: List[str] = config.get("ark_api_keys", [])
            self.ark_api_base: str = (config.get("ark_api_base") or "https://ark.cn-beijing.volces.com").strip()
            self.ark_model: str = (config.get("ark_model") or "doubao-seedream-4-0-250828").strip()
            self.ark_response_format: str = (config.get("ark_response_format") or "url").strip()
            self.ark_size: str = (config.get("ark_size") or "2K").strip()
            self.ark_stream: bool = bool(config.get("ark_stream", False))
            self.ark_watermark: bool = bool(config.get("ark_watermark", True))
            self.ark_sequential: str = (config.get("ark_sequential", "auto") or "auto").strip()
            try:
                self.ark_max_images: int = int(config.get("ark_max_images", 1) or 1)
            except Exception:
                self.ark_max_images = 1
            
            # 模型配置
            self.model_name: str = config.get("model_name", Constants.DEFAULT_MODEL_NAME).strip()
            if not self.model_name:
                raise ConfigError("模型名称不能为空")
            
            # 重试配置
            self.max_retry_attempts: int = config.get("max_retry_attempts", Constants.DEFAULT_MAX_RETRY_ATTEMPTS)
            if self.max_retry_attempts < 0:
                logger.warning("最大重试次数不能为负数，使用默认值")
                self.max_retry_attempts = Constants.DEFAULT_MAX_RETRY_ATTEMPTS
            
            self.nap_server_address: Optional[str] = config.get("nap_server_address")
            self.nap_server_port: Optional[int] = config.get("nap_server_port")

            # 每群每分钟调用次数限制（<=0 表示不限制）
            rate_limit_raw = config.get("calls_per_minute_per_group", Constants.DEFAULT_CALLS_PER_MINUTE_PER_GROUP)
            self.calls_per_minute_per_group: int = int(rate_limit_raw or Constants.DEFAULT_CALLS_PER_MINUTE_PER_GROUP)
            if self.calls_per_minute_per_group < 0:
                logger.warning("频率限制不能为负数，使用默认值")
                self.calls_per_minute_per_group = Constants.DEFAULT_CALLS_PER_MINUTE_PER_GROUP
                
            self._rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)  # key -> deque[timestamps]
            self._rate_lock: asyncio.Lock = asyncio.Lock()

            # QQ群访问控制配置
            self.group_access_mode: str = config.get("group_access_mode", "disabled").lower().strip()
            if self.group_access_mode not in ["disabled", "whitelist", "blacklist"]:
                logger.warning(f"无效的群访问控制模式: {self.group_access_mode}，使用默认值 disabled")
                self.group_access_mode = "disabled"
            
            # 群访问控制名单，确保为字符串列表
            group_list_raw = config.get("group_access_list", [])
            self.group_access_list: List[str] = []
            if isinstance(group_list_raw, list):
                for group_id in group_list_raw:
                    if isinstance(group_id, (str, int)):
                        self.group_access_list.append(str(group_id).strip())
            
            # 配置加载日志
            api_key_count = len(self.openrouter_api_keys)
            group_count = len(self.group_access_list) if self.group_access_mode != "disabled" else 0
            logger.info(f"配置加载完成 - Provider: {self.provider}, OpenRouter密钥: {api_key_count}, 模型: {self.model_name}, 频率限制: {self.calls_per_minute_per_group}次/分钟")
            logger.info(f"群访问控制: {self.group_access_mode}, 名单数量: {group_count}")
            if self.provider == "ark":
                logger.info(f"Ark 配置: base={self.ark_api_base}, model={self.ark_model}, response_format={self.ark_response_format}, size={self.ark_size}, stream={self.ark_stream}, watermark={self.ark_watermark}, seq={self.ark_sequential}, max_images={self.ark_max_images}, keys={len(self.ark_api_keys)}")
            logger.info("插件初始化完成")
            
        except (ValueError, TypeError) as e:
            logger.error(f"配置参数类型错误: {e}")
            raise ConfigError(Constants.ERROR_MSG_CONFIG_ERROR.format(str(e)))
        except Exception as e:
            logger.error(f"插件初始化失败: {e}")
            raise ConfigError(Constants.ERROR_MSG_CONFIG_ERROR.format(str(e)))

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

    def _is_group_allowed(self, event: AstrMessageEvent) -> Tuple[bool, str, bool]:
        """
        检查当前群是否有权限使用插件
        
        Args:
            event: 消息事件对象
            
        Returns:
            Tuple[bool, str, bool]: (是否允许, 错误消息, 是否静默)
                - 是否允许: True/False
                - 错误消息: 给用户的提示信息
                - 是否静默: True表示静默退出不响应，False表示返回错误消息
        """
        # 如果禁用了群访问控制，则允许所有群使用
        if self.group_access_mode == "disabled":
            return True, "", False
        
        # 获取群ID
        group_id = None
        try:
            group_id = event.get_group_id()
        except Exception:
            # 如果不是群聊（私聊等），则允许使用
            return True, "", False
        
        if not group_id:
            # 不是群聊，允许使用
            return True, "", False
        
        group_id_str = str(group_id).strip()
        
        # 白名单模式：只有在名单中的群才能使用
        if self.group_access_mode == "whitelist":
            if group_id_str in self.group_access_list:
                logger.debug(f"群 {group_id_str} 在白名单中，允许使用")
                return True, "", False
            else:
                logger.info(f"群 {group_id_str} 不在白名单中，拒绝使用")
                return False, Constants.ERROR_MSG_GROUP_NOT_ALLOWED, False
        
        # 黑名单模式：在名单中的群不能使用（静默）
        elif self.group_access_mode == "blacklist":
            if group_id_str in self.group_access_list:
                logger.info(f"群 {group_id_str} 在黑名单中，静默拒绝使用")
                return False, "", True  # 静默模式，不返回任何消息
            else:
                logger.debug(f"群 {group_id_str} 不在黑名单中，允许使用")
                return True, "", False
        
        # 未知模式，默认允许
        return True, "", False

    async def _try_acquire_rate(self, event: AstrMessageEvent) -> Tuple[bool, int, int]:
        """尝试消耗一次配额。返回 (allowed: bool, wait_seconds: int, remaining: int)."""
        limit = self.calls_per_minute_per_group
        if limit <= 0:
            return True, 0, -1
        key = self._group_key(event)
        now = time.time()
        async with self._rate_lock:
            bucket = self._rate_buckets[key]
            # 清理旧记录
            while bucket and (now - bucket[0]) >= Constants.RATE_LIMIT_WINDOW_SECONDS:
                bucket.popleft()
            if len(bucket) >= limit:
                oldest = bucket[0]
                wait = ceil(max(0, Constants.RATE_LIMIT_WINDOW_SECONDS - (now - oldest)))
                logger.warning(f"频率限制触发 - {key}, 当前调用次数: {len(bucket)}/{limit}, 需等待{wait}秒")
                return False, max(wait, Constants.MIN_WAIT_SECONDS), 0
            else:
                bucket.append(now)
                remaining = max(0, limit - len(bucket))
                logger.debug(f"频率检查通过 - {key}, 剩余配额: {remaining}/{limit}")
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
            # 兼容性增强：优先尝试以 base64 方式发送，避免 file:/// 在异构部署下无法访问
            try:
                async with aiofiles.open(image_path, "rb") as f:
                    img_bytes = await f.read()
                b64 = base64.b64encode(img_bytes).decode()
                return Image.fromBase64(b64)
            except Exception as e:
                logger.warning(f"base64 发送失败，退回本地文件发送: {e}")
                return Image.fromFileSystem(image_path)

        logger.info(f"检测到配置了callback_api_base: {callback_api_base}")
        try:
            image_component = Image.fromFileSystem(image_path)
            download_url = await image_component.register_to_file_service()
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

    async def _generate_image(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True) -> AsyncGenerator[Any, None]:
        """
        内部图像生成方法
        
        Args:
            event: 消息事件对象
            image_description: 图像描述
            use_reference_images: 是否使用参考图片
        """
        # 业务流程开始日志和进度提示
        logger.info(f"开始图像生成任务，描述长度: {len(image_description)}, 使用参考图片: {use_reference_images}")
        
        # 发送进度提示
        progress_msg = [Plain("🎨 正在生成图片，请稍候...")]
        yield event.chain_result(progress_msg)
        
        openrouter_api_keys = self.openrouter_api_keys
        nap_server_address = self.nap_server_address
        nap_server_port = self.nap_server_port

        # 根据参数决定是否使用参考图片
        input_images = []  # base64 列表（用于 OpenRouter）
        ark_image_urls: List[str] = []  # URL 列表（用于 Ark）
        if use_reference_images:
            # 从当前对话上下文中获取图片信息
            if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
                for comp in event.message_obj.message:
                    if isinstance(comp, Image):
                        try:
                            # OpenRouter 使用 base64；Ark 可尝试直接 URL（若可转换）
                            base64_data = await comp.convert_to_base64()
                            input_images.append(base64_data)
                            # Ark 引用图：优先使用 callback 文件服务 URL；退化为组件内置 http url
                            try:
                                url = None
                                # 优先注册到回调文件服务
                                url = await comp.register_to_file_service()
                                if url:
                                    ark_image_urls.append(url)
                                else:
                                    raise Exception("no url from register_to_file_service")
                            except Exception:
                                # 回退：若 comp.url 是 http 则使用
                                try:
                                    if getattr(comp, 'url', '') and str(comp.url).startswith("http"):
                                        ark_image_urls.append(comp.url)
                                except Exception:
                                    pass
                        except (IOError, ValueError, OSError) as e:
                            logger.warning(f"转换当前消息中的参考图片到base64失败: {e}")
                        except Exception as e:
                            logger.error(f"处理当前消息中的图片时出现未预期的错误: {e}")
                    elif isinstance(comp, Reply):
                        if comp.chain:
                            for reply_comp in comp.chain:
                                if isinstance(reply_comp, Image):
                                    try:
                                        base64_data = await reply_comp.convert_to_base64()
                                        input_images.append(base64_data)
                                        try:
                                            url = await reply_comp.register_to_file_service()
                                            if url:
                                                ark_image_urls.append(url)
                                            else:
                                                raise Exception("no url from register_to_file_service")
                                        except Exception:
                                            try:
                                                if getattr(reply_comp, 'url', '') and str(reply_comp.url).startswith("http"):
                                                    ark_image_urls.append(reply_comp.url)
                                            except Exception:
                                                pass
                                        logger.info(f"从引用消息中获取到图片")
                                    except (IOError, ValueError, OSError) as e:
                                        logger.warning(f"转换引用消息中的参考图片到base64失败: {e}")
                                    except Exception as e:
                                        logger.error(f"处理引用消息中的图片时出现未预期的错误: {e}")
                        else:
                            logger.debug("引用消息的chain为空，无法获取图片内容")

            if input_images:
                logger.info(f"使用了 {len(input_images)} 张参考图片进行图像生成")
            else:
                logger.info("未找到参考图片，执行纯文本图像生成")

        # 调用生成图像的函数（根据 provider 分支）
        try:
            if self.provider == "ark":
                if not self.ark_api_keys:
                    yield event.plain_result(Constants.ERROR_MSG_ARK_NO_KEY)
                    return
                image_url, image_path = await generate_image_ark(
                    image_description,
                    self.ark_api_keys,
                    model=self.ark_model,
                    image_urls=ark_image_urls,
                    api_base=self.ark_api_base,
                    response_format=self.ark_response_format,
                    size=self.ark_size,
                    stream=self.ark_stream,
                    watermark=self.ark_watermark,
                    sequential_image_generation=self.ark_sequential,
                    max_images=self.ark_max_images,
                    max_retry_attempts=self.max_retry_attempts
                )
            else:
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
                error_chain = [Plain(Constants.ERROR_MSG_GENERATION_FAILED)]
                yield event.chain_result(error_chain)
                return
            
            # 处理文件传输和图片发送
            if self.nap_server_address and self.nap_server_address != "localhost":
                image_path = await send_file(image_path, host=nap_server_address, port=nap_server_port)
            
            # 使用新的发送方法，优先使用callback_api_base
            image_component = await self.send_image_with_callback_api(image_path)
            chain = [image_component]
            logger.info("图像生成任务完成")
            yield event.chain_result(chain)
            return
                
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"网络连接错误导致图像生成失败: {e}")
            error_chain = [Plain(Constants.ERROR_MSG_NETWORK_ERROR.format(str(e)))]
            yield event.chain_result(error_chain)
            return
        except ValueError as e:
            logger.error(f"参数错误导致图像生成失败: {e}")
            error_chain = [Plain(Constants.ERROR_MSG_PARAM_ERROR.format(str(e)))]
            yield event.chain_result(error_chain)
            return
        except Exception as e:
            logger.error(f"图像生成过程出现未预期的错误: {e}")
            error_chain = [Plain(Constants.ERROR_MSG_UNKNOWN_ERROR.format(str(e)))]
            yield event.chain_result(error_chain)
            return

    # 通过指令触发：/aiimg <描述>
    @filter.command("aiimg")
    async def aiimg(self, event: AstrMessageEvent, prompt: str = "") -> AsyncGenerator[Any, None]:
        """使用命令触发图片生成。示例：/aiimg 画一只可爱的猫咪"""
        # 记录指令触发日志
        try:
            user_name = event.get_sender_name()
        except:
            user_name = "未知用户"
        logger.info(f"用户{user_name}触发/aiimg指令")
        
        # 禁用默认 LLM 自动回复，避免与指令输出冲突
        try:
            event.should_call_llm(False)
        except Exception:
            pass

        # 检查群访问权限
        group_allowed, error_msg, is_silent = self._is_group_allowed(event)
        if not group_allowed:
            if is_silent:
                # 静默模式，直接退出不响应
                return
            else:
                # 非静默模式，返回错误消息
                yield event.plain_result(error_msg)
                return

        allowed, wait, remaining = await self._try_acquire_rate(event)
        if not allowed:
            yield event.plain_result(
                Constants.ERROR_MSG_RATE_LIMIT.format(self.calls_per_minute_per_group, wait)
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
            yield event.plain_result(Constants.ERROR_MSG_NO_DESCRIPTION)
            return

        # 智能检测是否有参考图片可用
        has_reference_images = False
        try:
            if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
                for comp in event.message_obj.message:
                    if isinstance(comp, Image):
                        has_reference_images = True
                        break
                    elif isinstance(comp, Reply) and getattr(comp, 'chain', None):
                        for reply_comp in comp.chain:
                            if isinstance(reply_comp, Image):
                                has_reference_images = True
                                break
                    if has_reference_images:
                        break
        except Exception:
            pass

        # 根据实际情况决定是否使用参考图片
        async for result in self._generate_image(event, image_description=desc, use_reference_images=has_reference_images):
            yield result

    # 通过指令触发：/aiimg手办化
    @filter.command("aiimg手办化")
    async def aiimg_shouban(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """将参考图片"手办化"。用法：发送图片并输入 /aiimg手办化，或引用含图片的消息再输入指令。"""
        # 记录指令触发日志
        try:
            user_name = event.get_sender_name()
        except:
            user_name = "未知用户"
        logger.info(f"用户{user_name}触发/aiimg手办化指令")
        
        # 禁用默认 LLM 自动回复
        try:
            event.should_call_llm(False)
        except Exception:
            pass

        # 检查群访问权限
        group_allowed, error_msg, is_silent = self._is_group_allowed(event)
        if not group_allowed:
            if is_silent:
                # 静默模式，直接退出不响应
                return
            else:
                # 非静默模式，返回错误消息
                yield event.plain_result(error_msg)
                return

        allowed, wait, remaining = await self._try_acquire_rate(event)
        if not allowed:
            yield event.plain_result(
                Constants.ERROR_MSG_RATE_LIMIT.format(self.calls_per_minute_per_group, wait)
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
            yield event.plain_result(Constants.ERROR_MSG_NO_IMAGE)
            return

        # 使用常量中定义的手办化固定提示词
        async for result in self._generate_image(event, image_description=Constants.SHOUBAN_PROMPT, use_reference_images=True):
            yield result

    # 通过指令触发：/aiimg手办化2
    @filter.command("aiimg手办化2")
    async def aiimg_shouban2(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """将参考图片"手办化2"（英文版本）。用法：发送图片并输入 /aiimg手办化2，或引用含图片的消息再输入指令。"""
        # 记录指令触发日志
        try:
            user_name = event.get_sender_name()
        except:
            user_name = "未知用户"
        logger.info(f"用户{user_name}触发/aiimg手办化2指令")
        
        # 禁用默认 LLM 自动回复
        try:
            event.should_call_llm(False)
        except Exception:
            pass

        # 检查群访问权限
        group_allowed, error_msg, is_silent = self._is_group_allowed(event)
        if not group_allowed:
            if is_silent:
                # 静默模式，直接退出不响应
                return
            else:
                # 非静默模式，返回错误消息
                yield event.plain_result(error_msg)
                return

        allowed, wait, remaining = await self._try_acquire_rate(event)
        if not allowed:
            yield event.plain_result(
                Constants.ERROR_MSG_RATE_LIMIT.format(self.calls_per_minute_per_group, wait)
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
            yield event.plain_result("请先发送一张图片，或引用包含图片的消息后再使用指令：/aiimg手办化2")
            return

        # 使用常量中定义的手办化2固定提示词（英文版本）
        async for result in self._generate_image(event, image_description=Constants.SHOUBAN2_PROMPT, use_reference_images=True):
            yield result

    # 通过指令触发：/aiimg帮助
    @filter.command("aiimg帮助")
    async def aiimg_help(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """显示插件帮助信息"""
        # 记录指令触发日志
        try:
            user_name = event.get_sender_name()
        except:
            user_name = "未知用户"
        logger.info(f"用户{user_name}触发/aiimg帮助指令")
        
        # 禁用默认 LLM 自动回复
        try:
            event.should_call_llm(False)
        except Exception:
            pass

        # 检查群访问权限
        group_allowed, error_msg, is_silent = self._is_group_allowed(event)
        if not group_allowed:
            if is_silent:
                # 静默模式，直接退出不响应
                return
            else:
                # 非静默模式，返回错误消息
                yield event.plain_result(error_msg)
                return
        
        # 直接返回帮助信息，不消耗API配额
        yield event.plain_result(Constants.HELP_MESSAGE)
