import random
import aiohttp
import asyncio
import aiofiles
import base64
import os
import re
import uuid
from datetime import datetime, timedelta
import glob
from pathlib import Path
from astrbot.api import logger
from astrbot.api.star import StarTools


class ImageGeneratorState:
    """图像生成器状态管理类，用于处理并发安全"""
    def __init__(self):
        self.last_saved_image = {"url": None, "path": None}
        self.api_key_index = 0
        self._lock = asyncio.Lock()
    
    async def get_next_api_key(self, api_keys):
        """获取下一个可用的API密钥"""
        async with self._lock:
            if not api_keys or not isinstance(api_keys, list):
                raise ValueError("API密钥列表不能为空")
            current_key = api_keys[self.api_key_index % len(api_keys)]
            return current_key
    
    async def rotate_to_next_api_key(self, api_keys):
        """轮换到下一个API密钥"""
        async with self._lock:
            if api_keys and isinstance(api_keys, list) and len(api_keys) > 1:
                self.api_key_index = (self.api_key_index + 1) % len(api_keys)
                logger.info(f"已轮换到下一个API密钥，当前索引: {self.api_key_index}")
    
    async def update_saved_image(self, url, path):
        """更新保存的图像信息"""
        async with self._lock:
            self.last_saved_image = {"url": url, "path": path}
    
    async def get_saved_image_info(self):
        """获取最后保存的图像信息"""
        async with self._lock:
            return self.last_saved_image["url"], self.last_saved_image["path"]


# 全局状态管理实例
_state = ImageGeneratorState()


async def cleanup_old_images(data_dir=None):
    """
    清理超过15分钟的图像文件
    
    Args:
        data_dir (Path): 数据目录路径，如果为None则使用当前脚本目录
    """
    try:
        # 如果没有传入data_dir，使用当前脚本目录
        if data_dir is None:
            script_dir = Path(__file__).parent.parent
            data_dir = script_dir
        
        images_dir = data_dir / "images"

        if not images_dir.exists():
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=15)

        # 查找images目录下的所有图像文件
        image_patterns = ["gemini_image_*.png", "gemini_image_*.jpg", "gemini_image_*.jpeg"]

        for pattern in image_patterns:
            for file_path in images_dir.glob(pattern):
                try:
                    # 获取文件的修改时间
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    # 如果文件超过15分钟，删除它
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        logger.info(f"已清理过期图像: {file_path}")

                except Exception as e:
                    logger.warning(f"清理文件 {file_path} 时出错: {e}")

    except Exception as e:
        logger.error(f"图像清理过程出错: {e}")


async def save_base64_image(base64_string, image_format="png", data_dir=None):
    """
    保存base64图像数据到images文件夹

    Args:
        base64_string (str): base64编码的图像数据
        image_format (str): 图像格式
        data_dir (Path): 数据目录路径，如果为None则使用当前脚本目录

    Returns:
        bool: 是否保存成功
    """
    try:
        # 如果没有传入data_dir，使用当前脚本目录
        if data_dir is None:
            script_dir = Path(__file__).parent.parent
            data_dir = script_dir
        
        images_dir = data_dir / "images"
        # 确保images目录存在
        images_dir.mkdir(exist_ok=True)
        
        # 先清理旧图像
        await cleanup_old_images(data_dir)

        # 解码 base64 数据
        image_data = base64.b64decode(base64_string)

        # 生成唯一文件名（使用时间戳和UUID避免冲突）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        image_path = images_dir / f"gemini_image_{timestamp}_{unique_id}.{image_format}"

        # 保存图像文件
        async with aiofiles.open(image_path, "wb") as f:
            await f.write(image_data)

        # 获取绝对路径
        abs_path = str(image_path.absolute())
        file_url = f"file://{abs_path}"

        # 更新状态
        await _state.update_saved_image(file_url, str(image_path))

        logger.info(f"图像已保存到: {abs_path}")
        logger.debug(f"文件大小: {len(image_data)} bytes")

        return True

    except base64.binascii.Error as e:
        logger.error(f"Base64 解码失败: {e}")
        return False
    except Exception as e:
        logger.error(f"保存图像文件失败: {e}")
        return False


async def get_next_api_key(api_keys):
    """
    获取下一个可用的API密钥
    
    Args:
        api_keys (list): API密钥列表
        
    Returns:
        str: 当前可用的API密钥
    """
    return await _state.get_next_api_key(api_keys)


async def rotate_to_next_api_key(api_keys):
    """
    轮换到下一个API密钥
    
    Args:
        api_keys (list): API密钥列表
    """
    await _state.rotate_to_next_api_key(api_keys)


async def get_saved_image_info():
    """
    获取最后保存的图像信息

    Returns:
        tuple: (image_url, image_path)
    """
    return await _state.get_saved_image_info()


async def generate_image_openrouter(prompt, api_keys, model="google/gemini-2.5-flash-image-preview:free", max_tokens=1000, input_images=None, api_base=None, max_retry_attempts=3):
    """
    Generate image using OpenRouter API with Gemini model, supports multiple API keys with automatic rotation and retry mechanism

    Args:
        prompt (str): The prompt for image generation
        api_keys (list): List of OpenRouter API keys for rotation
        model (str): Model to use (default: google/gemini-2.5-flash-image-preview:free)
        max_tokens (int): Maximum tokens for the response
        input_images (list): List of base64 encoded input images (optional)
        api_base (str): Custom API base URL (optional, defaults to OpenRouter)
        max_retry_attempts (int): Maximum number of retry attempts per API key (default: 3)

    Returns:
        tuple: (image_url, image_path) or (None, None) if failed
    """
    # 兼容性处理：如果传入单个API密钥字符串，转换为列表
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    
    if not api_keys:
        logger.error("未提供API密钥")
        return None, None
    
    # 支持自定义API base
    if api_base:
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
    
    # 尝试每个API密钥，对每个密钥进行重试
    max_api_attempts = len(api_keys)
    
    for api_attempt in range(max_api_attempts):
        try:
            current_api_key = await get_next_api_key(api_keys)
            current_index = (_state.api_key_index % len(api_keys)) + 1
            
            # 对当前API密钥进行多次重试
            for retry_attempt in range(max_retry_attempts):
                try:
                    if retry_attempt > 0:
                        # 重试时的延迟，指数退避
                        delay = min(2 ** retry_attempt, 10)
                        logger.info(f"API密钥 #{current_index} 重试 {retry_attempt + 1}/{max_retry_attempts}，等待 {delay} 秒...")
                        await asyncio.sleep(delay)
                    else:
                        logger.info(f"尝试使用API密钥 #{current_index}")
                    
                    # 构建消息内容，支持输入图片
                    message_content = []
                    
                    # 添加文本内容
                    message_content.append({
                        "type": "text",
                        "text": f"Generate an image: {prompt}"
                    })
                    
                    # 如果有输入图片，添加到消息中
                    if input_images:
                        for base64_image in input_images:
                            # 确保base64数据包含正确的data URI格式
                            if not base64_image.startswith('data:image/'):
                                # 假设是PNG格式，添加data URI前缀
                                base64_image = f"data:image/png;base64,{base64_image}"
                            
                            message_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                }
                            })

                    # 为 Gemini 图像生成构建payload
                    payload = {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": message_content if len(message_content) > 1 else f"Generate an image: {prompt}"
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    }

                    headers = {
                        "Authorization": f"Bearer {current_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/astrbot",
                        "X-Title": "AstrBot LLM Draw Plus"
                    }

                    # 调试输出：打印请求结构
                    if retry_attempt == 0:  # 只在第一次尝试时打印调试信息
                        logger.debug(f"模型: {model}")
                        logger.debug(f"输入图片数量: {len(input_images) if input_images else 0}")
                        if input_images:
                            logger.debug(f"第一张图片base64长度: {len(input_images[0])}")
                        logger.debug(f"消息内容结构: {type(payload['messages'][0]['content'])}")
                        if isinstance(payload['messages'][0]['content'], list):
                            content_types = [item.get('type', 'unknown') for item in payload['messages'][0]['content']]
                            logger.debug(f"消息内容类型: {content_types}")

                    timeout = aiohttp.ClientTimeout(total=60)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, json=payload, headers=headers) as response:
                            data = await response.json()
                            
                            if retry_attempt == 0:  # 只在第一次尝试时打印详细调试信息
                                logger.debug(f"API响应状态: {response.status}")
                                logger.debug(f"响应数据键: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")

                            if response.status == 200 and "choices" in data:
                                choice = data["choices"][0]
                                message = choice["message"]
                                content = message["content"]

                                # 检查 Gemini 标准的 message.images 字段
                                if "images" in message and message["images"]:
                                    logger.info(f"Gemini 返回了 {len(message['images'])} 个图像")

                                    for i, image_item in enumerate(message["images"]):
                                        if "image_url" in image_item and "url" in image_item["image_url"]:
                                            image_url = image_item["image_url"]["url"]

                                            # 检查是否是 base64 格式
                                            if image_url.startswith("data:image/"):
                                                try:
                                                    # 解析 data URI: data:image/png;base64,iVBORw0KGg...
                                                    header, base64_data = image_url.split(",", 1)
                                                    image_format = header.split("/")[1].split(";")[0]

                                                    if await save_base64_image(base64_data, image_format):
                                                        logger.info(f"API密钥 #{current_index} 成功生成图像")
                                                        return await get_saved_image_info()

                                                except Exception as e:
                                                    logger.warning(f"解析图像 {i+1} 失败: {e}")
                                                    continue

                                # 如果没有找到标准images字段，尝试在content中查找
                                elif isinstance(content, str):
                                    # 查找内联的 base64 图像数据
                                    base64_pattern = r"data:image/([^;]+);base64,([A-Za-z0-9+/=]+)"
                                    matches = re.findall(base64_pattern, content)

                                    if matches:
                                        image_format, base64_string = matches[0]
                                        if await save_base64_image(base64_string, image_format):
                                            logger.info(f"API密钥 #{current_index} 成功生成图像")
                                            return await get_saved_image_info()

                                logger.info("API调用成功，但未找到图像数据")
                                # 这种情况也算成功，不需要重试
                                return None, None

                            elif response.status == 429 or (response.status == 402 and "insufficient" in str(data).lower()):
                                # 额度耗尽或速率限制，直接尝试下一个密钥，不进行重试
                                error_msg = data.get("error", {}).get("message", f"HTTP {response.status}")
                                logger.warning(f"API密钥 #{current_index} 额度耗尽或速率限制: {error_msg}")
                                break  # 跳出重试循环，尝试下一个API密钥
                            else:
                                # 其他错误，可以重试
                                error_msg = data.get("error", {}).get("message", f"HTTP {response.status}")
                                logger.warning(f"OpenRouter API 错误 (重试 {retry_attempt + 1}/{max_retry_attempts}): {error_msg}")
                                if "error" in data:
                                    logger.debug(f"完整错误信息: {data['error']}")
                                
                                if retry_attempt == max_retry_attempts - 1:
                                    logger.error(f"API密钥 #{current_index} 达到最大重试次数")
                                    break  # 跳出重试循环，尝试下一个API密钥

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"网络请求失败 (密钥 #{current_index}, 重试 {retry_attempt + 1}/{max_retry_attempts}): {str(e)}")
                    if retry_attempt == max_retry_attempts - 1:
                        logger.error(f"API密钥 #{current_index} 网络连接达到最大重试次数")
                        break  # 跳出重试循环，尝试下一个API密钥
                except Exception as e:
                    logger.error(f"调用 OpenRouter API 时发生异常 (密钥 #{current_index}, 重试 {retry_attempt + 1}/{max_retry_attempts}): {str(e)}")
                    if retry_attempt == max_retry_attempts - 1:
                        logger.error(f"API密钥 #{current_index} 异常达到最大重试次数")
                        break  # 跳出重试循环，尝试下一个API密钥
        
        except Exception as e:
            logger.error(f"处理API密钥 #{current_index} 时发生异常: {str(e)}")
        
        # 尝试下一个API密钥
        if api_attempt < max_api_attempts - 1:
            await rotate_to_next_api_key(api_keys)
            logger.info(f"切换到下一个API密钥")
    
    logger.error("所有API密钥和重试次数已耗尽")
    return None, None


async def generate_image(prompt, api_key, model="stabilityai/stable-diffusion-3-5-large", seed=None, image_size="1024x1024"):
    """
    生成图像使用SiliconFlow API
    
    Args:
        prompt (str): 图像生成提示
        api_key (str): API密钥
        model (str): 模型名称
        seed (int): 随机种子
        image_size (str): 图像尺寸
        
    Returns:
        tuple: (image_url, image_path) or (None, None) if failed
    """
    url = "https://api.siliconflow.cn/v1/images/generations"

    if seed is None:
        seed = random.randint(0, 9999999999)

    payload = {
        "model": model,
        "prompt": prompt,
        "image_size": image_size,
        "seed": seed
    }
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"
    }

    max_retries = 10  # 最大重试次数
    retry_count = 0
    
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while retry_count < max_retries:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    data = await response.json()

                    if data.get("code") == 50603:
                        logger.warning("系统繁忙，1秒后重试")
                        await asyncio.sleep(1)
                        retry_count += 1
                        continue

                    if "images" in data:
                        for image in data["images"]:
                            image_url = image["url"]
                            async with session.get(image_url) as img_response:
                                if img_response.status == 200:
                                    # 生成唯一文件名
                                    script_dir = Path(__file__).parent.parent
                                    images_dir = script_dir / "images"
                                    images_dir.mkdir(exist_ok=True)
                                    
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    unique_id = str(uuid.uuid4())[:8]
                                    image_path = images_dir / f"siliconflow_image_{timestamp}_{unique_id}.jpeg"
                                    
                                    async with aiofiles.open(image_path, "wb") as f:
                                        await f.write(await img_response.read())
                                    
                                    logger.info(f"图像已下载: {image_url} -> {image_path}")
                                    return image_url, str(image_path)
                                else:
                                    logger.error(f"下载图像失败: {image_url}")
                                    return None, None
                    else:
                        logger.warning("响应中未找到图像")
                        return None, None
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"网络请求失败 (重试 {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # 指数退避
                else:
                    return None, None
                    
    logger.error(f"达到最大重试次数 ({max_retries})，生成失败")
    return None, None


async def _download_image_to_file(session, image_url, prefix="ark_image"):
    try:
        async with session.get(image_url) as resp:
            if resp.status != 200:
                logger.error(f"下载图像失败: HTTP {resp.status} - {image_url}")
                return None
            content_type = resp.headers.get("Content-Type", "image/png")
            # 猜测扩展名
            if "/" in content_type:
                ext = content_type.split("/")[-1].split(";")[0].strip()
                if ext in ("jpeg", "jpg", "png", "webp", "gif"):
                    image_ext = "jpg" if ext == "jpeg" else ext
                else:
                    image_ext = "png"
            else:
                image_ext = "png"

            script_dir = Path(__file__).parent.parent
            images_dir = script_dir / "images"
            images_dir.mkdir(exist_ok=True)
            await cleanup_old_images(script_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            image_path = images_dir / f"{prefix}_{timestamp}_{unique_id}.{image_ext}"

            async with aiofiles.open(image_path, "wb") as f:
                await f.write(await resp.read())

            await _state.update_saved_image(image_url, str(image_path))
            logger.info(f"图像已下载: {image_url} -> {image_path}")
            return str(image_path)
    except Exception as e:
        logger.error(f"下载图像异常: {e}")
        return None


async def generate_image_ark(
    prompt,
    api_keys,
    model="doubao-seedream-4-0-250828",
    image_urls=None,
    api_base="https://ark.cn-beijing.volces.com",
    response_format="url",
    size="2K",
    stream=False,
    watermark=True,
    sequential_image_generation="auto",
    max_images=1,
    max_retry_attempts=3
):
    """
    Generate images via ByteDance Ark Images API (v3) with key rotation and retries.

    Returns: tuple (first_image_url, first_image_local_path) or (None, None)
    """
    # 兼容传入单Key
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    if not api_keys:
        logger.error("未提供 Ark API 密钥")
        return None, None

    url = f"{api_base.rstrip('/')}/api/v3/images/generations"
    image_urls = image_urls or []

    # 每个密钥进行多次重试
    for api_attempt in range(len(api_keys)):
        try:
            current_api_key = await get_next_api_key(api_keys)
            current_index = (_state.api_key_index % len(api_keys)) + 1

            for retry_attempt in range(max_retry_attempts):
                try:
                    if retry_attempt > 0:
                        delay = min(2 ** retry_attempt, 10)
                        logger.info(f"Ark 密钥 #{current_index} 重试 {retry_attempt + 1}/{max_retry_attempts}，等待 {delay} 秒...")
                        await asyncio.sleep(delay)
                    else:
                        logger.info(f"尝试使用 Ark 密钥 #{current_index}")

                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "response_format": response_format,
                        "size": size,
                        "watermark": bool(watermark),
                        # Ark 示例字段
                        "sequential_image_generation": sequential_image_generation,
                        "sequential_image_generation_options": {
                            "max_images": int(max_images) if max_images else 1
                        }
                    }
                    if image_urls:
                        payload["image"] = image_urls
                    # Ark 支持 stream，但此处作为普通请求处理
                    if stream:
                        payload["stream"] = True

                    headers = {
                        "Authorization": f"Bearer {current_api_key}",
                        "Content-Type": "application/json"
                    }

                    timeout = aiohttp.ClientTimeout(total=120)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, json=payload, headers=headers) as response:
                            # 优先尝试解析 JSON
                            data = None
                            text_body = await response.text()
                            try:
                                data = await response.json()
                            except Exception:
                                # 不是纯 JSON，可能是流或错误文本
                                data = None

                            if response.status == 200:
                                # 尝试从 data 或文本中提取 URL 列表
                                urls = []
                                if isinstance(data, dict):
                                    # 常见结构尝试
                                    if isinstance(data.get("data"), list):
                                        for item in data["data"]:
                                            if isinstance(item, dict) and "url" in item:
                                                urls.append(item["url"])
                                    elif isinstance(data.get("images"), list):
                                        for item in data["images"]:
                                            if isinstance(item, dict) and "url" in item:
                                                urls.append(item["url"])
                                            elif isinstance(item, str):
                                                urls.append(item)
                                    elif isinstance(data.get("output"), dict):
                                        out = data["output"]
                                        if isinstance(out.get("images"), list):
                                            for u in out["images"]:
                                                if isinstance(u, str):
                                                    urls.append(u)
                                # 兜底：从文本里用正则提取 http(s) URL
                                if not urls and text_body:
                                    url_pattern = r"https?://[^\s\"]+"
                                    import re as _re
                                    urls = _re.findall(url_pattern, text_body) or []

                                if not urls:
                                    logger.info("Ark API 调用成功，但未找到图像 URL")
                                    return None, None

                                # 下载首张图像保存
                                first_url = urls[0]
                                saved_path = await _download_image_to_file(session, first_url, prefix="ark_image")
                                if saved_path:
                                    return first_url, saved_path
                                else:
                                    # 下载失败不再重试当前响应
                                    return None, None

                            elif response.status in (429, 403):
                                logger.warning(f"Ark 密钥 #{current_index} 额度或速率限制: HTTP {response.status}")
                                break  # 切换密钥
                            else:
                                logger.warning(f"Ark API 错误 (重试 {retry_attempt + 1}/{max_retry_attempts}): HTTP {response.status} {text_body[:200]}")
                                if retry_attempt == max_retry_attempts - 1:
                                    logger.error(f"Ark 密钥 #{current_index} 达到最大重试次数")
                                    break

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Ark 网络请求失败 (密钥 #{current_index}, 重试 {retry_attempt + 1}/{max_retry_attempts}): {str(e)}")
                    if retry_attempt == max_retry_attempts - 1:
                        logger.error(f"Ark 密钥 #{current_index} 网络连接达到最大重试次数")
                        break
                except Exception as e:
                    logger.error(f"调用 Ark API 异常 (密钥 #{current_index}, 重试 {retry_attempt + 1}/{max_retry_attempts}): {str(e)}")
                    if retry_attempt == max_retry_attempts - 1:
                        logger.error(f"Ark 密钥 #{current_index} 异常达到最大重试次数")
                        break

        except Exception as e:
            logger.error(f"处理 Ark 密钥 #{current_index} 时发生异常: {str(e)}")

        # 轮换密钥
        if api_attempt < len(api_keys) - 1:
            await rotate_to_next_api_key(api_keys)
            logger.info("切换到下一个 Ark API 密钥")

    logger.error("所有 Ark API 密钥与重试已耗尽")
    return None, None


if __name__ == "__main__":
    async def create_test_image_base64():
        """创建一个测试用的小图片的base64数据"""
        import io
        from PIL import Image as PILImage, ImageDraw
        
        # 创建一个简单的测试图片
        img = PILImage.new('RGB', (100, 100), color='red')
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), "TEST", fill='white')
        
        # 转换为base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode()

    async def main():
        logger.info("测试 OpenRouter Gemini 图像生成...")
        # 从环境变量读取API密钥
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        if not openrouter_api_key:
            logger.error("请设置环境变量 OPENROUTER_API_KEY")
            return

        logger.info("\n=== 测试1: 先生成一张图片 ===")
        initial_prompt = "一只可爱的红色小熊猫，数字艺术风格"
        
        image_url, image_path = await generate_image_openrouter(
            initial_prompt,
            [openrouter_api_key],
            model="google/gemini-2.5-flash-image-preview:free"
        )
        
        if image_url and image_path:
            logger.info("初始图像生成成功!")
            logger.info(f"文件路径: {image_path}")
            
            logger.info("\n=== 测试2: 使用生成的图片进行修改 ===")
            try:
                # 读取刚生成的图片并转换为base64
                async with aiofiles.open(image_path, 'rb') as f:
                    image_bytes = await f.read()
                generated_image_base64 = base64.b64encode(image_bytes).decode()
                
                logger.info(f"生成图片的base64长度: {len(generated_image_base64)}")
                
                # 使用生成的图片进行修改
                modify_prompt = "将这张图片修改为蓝色主题，并添加一些星星装饰"
                input_images = [generated_image_base64]
                
                logger.info("正在使用生成的图片进行修改...")
                modified_url, modified_path = await generate_image_openrouter(
                    modify_prompt,
                    [openrouter_api_key],
                    model="google/gemini-2.5-flash-image-preview:free",
                    input_images=input_images
                )
                
                if modified_url and modified_path:
                    logger.info("图片修改成功!")
                    logger.info(f"修改后文件路径: {modified_path}")
                else:
                    logger.error("图片修改失败")
                    
            except Exception as e:
                logger.error(f"图片修改过程出错: {e}")
        else:
            logger.error("初始图像生成失败，无法进行后续修改测试")

        logger.info("\n=== 测试3: 检查多模态请求格式 ===")
        # 不实际发送请求，只检查构造的payload格式
        try:
            test_image_base64 = await create_test_image_base64()
            
            # 模拟构造请求，检查格式
            message_content = []
            message_content.append({
                "type": "text", 
                "text": f"Generate an image: {initial_prompt}"
            })
            
            base64_image = f"data:image/png;base64,{test_image_base64}"
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": base64_image
                }
            })
            
            payload = {
                "model": "google/gemini-2.5-flash-image-preview:free",
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            logger.info("多模态请求格式构造成功")
            logger.info(f"消息内容类型数量: {len(message_content)}")
            logger.info(f"包含文本: {any(item['type'] == 'text' for item in message_content)}")
            logger.info(f"包含图片: {any(item['type'] == 'image_url' for item in message_content)}")
            logger.info(f"图片URL前缀: {message_content[1]['image_url']['url'][:50]}...")
            
        except Exception as e:
            logger.error(f"请求格式检查出错: {e}")

    asyncio.run(main())
