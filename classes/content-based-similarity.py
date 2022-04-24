from transformers import T5Tokenizer, T5ForConditionalGeneration
from classes.helper import get_course

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

course_id = 2356382
df_course = get_course()
df_course.drop_duplicates(subset=['id'], inplace=True)

def similarity(sentence1, sentence2):
    input_ids = tokenizer.encode(f'stsb sentence1: {sentence1} sentence2: {sentence2} </s>', return_tensors='pt')

    outputs = model.generate(input_ids=input_ids)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return out

title = df_course.loc[df_course.id==course_id]['title']
print(title)

df_course[course_id] = df_course.apply(lambda x: similarity(title, x.title), axis=1)
print(df_course.sort_values(by=course_id))