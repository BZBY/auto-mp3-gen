#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频特色帧提取器 - 模块化版本
"""

from .core.extractor import KeyFrameExtractor
from .core.config import Config

__version__ = "2.0.0"
__author__ = "AI Assistant"

__all__ = [
    "KeyFrameExtractor",
    "Config",
]