from setuptools import setup, find_packages

setup(
    name="xtransformer",  # 包名
    version="0.3.0",  # 版本号
    author="Leoauro Xie",  # 作者
    author_email="xzcfly@qq.com",  # 作者邮箱
    description="transformer 实现，支持 decoder_only 和 encoder_only以及完整的transformer",  # 简短描述
    long_description=open("README.md").read(),  # 长描述（通常从 README.md 读取）
    long_description_content_type="text/markdown",  # 长描述格式
    url="https://github.com/Leoauro/xtransformer",  # 项目主页
    packages=find_packages(),  # 自动查找包
    install_requires=[
        "torch>=2.6.0",
        "setuptools>=70.2.0",
    ],  # 依赖列表
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Python 版本要求
)
