import re

def extract_text_between_stars(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式匹配双星号之间的文本
    pattern = r'\*\*(.*?)\*\*'
    matches = re.findall(pattern, content)

    # 返回匹配到的文本列表
    return matches

def extract_text_from_parentheses(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 正则表达式，匹配圆括号内的文本: 注意中文输入和英文输入是不同的
    pattern = r'\((.*?)\)'
    
    # 使用re.findall()方法找到所有匹配的文本
    matches = re.findall(pattern, content)
    
    # 返回匹配结果
    return matches


# 使用示例
file_path = 'content2.txt'  # 替换为你的文件路径
extracted_texts = extract_text_from_parentheses(file_path)
extracted_texts = extract_text_from_parentheses(file_path)
for text in extracted_texts:
    print(f"可以告诉我所有{text}分类下的单词吗")
