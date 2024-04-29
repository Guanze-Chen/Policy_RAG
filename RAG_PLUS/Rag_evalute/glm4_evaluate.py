import os 
import json
from tqdm import tqdm
from zhipuai import ZhipuAI
from langchain.schema import SystemMessage

client = ZhipuAI(api_key="sssk")
def ask_glm4(prompt):
    response = client.chat.completions.create(
        model='glm-4',
        messages=[{"role": "user", "content": prompt}]
    
    )
    return response.choices[0].message.content


EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria in Chinese}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""



def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    EVALUATION_PROMPT,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r", encoding='utf-8'))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = EVALUATION_PROMPT.format(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model(eval_prompt)
        feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w", encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False)

if __name__ == "__main__":
    import os 
    path = './RAG_output'
    os_list_dir = os.listdir(path)
    for file_name in os_list_dir:
        output_file_name = os.path.join(path, file_name)
        print(output_file_name)

        print("Running evaluation...")
        evaluate_answers(
                output_file_name,
                ask_glm4,
                'glm4',
                EVALUATION_PROMPT,
            )
