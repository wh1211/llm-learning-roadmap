from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")
res = classifier("这个产品非常好用，我很满意")
print(res)

# Qwen大模型文本生成
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device_map="cpu")
out = generator("我是一名软件工程学生，想学习大模型，请给我一点学习建议。", max_new_tokens=128)
print(out)
