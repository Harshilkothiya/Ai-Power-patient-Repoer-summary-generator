from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
print('Listing models...')
for m in genai.list_models():
    # m is a dict-like object, print model id and methods if present
    try:
        print(m)
    except Exception as e:
        print('Error printing model:', e)
