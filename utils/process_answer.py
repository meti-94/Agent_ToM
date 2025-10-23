import re
def get_cot_answer(answer):
    cot_answer = answer.split('Here are some examples:')[0].split('Question')[0].split('Final Answer:')[0].split('Please let me')[0].strip()
    if "The answer is:" in cot_answer and "Answer:" not in cot_answer:
        cot_answer = cot_answer.replace("The answer is:", "Answer:")
    if "The answer is:" in cot_answer and "Answer:" in cot_answer:
        cot_answer = cot_answer.split('The answer is:')[0].strip()
    return cot_answer

def get_clean_answer(args,cot_answer, allow_answer=False):
    if "Answer:" in cot_answer:
        if args.dataset == "strategyqa":
            split_answer = cot_answer.split('Answer:')[1].strip()
            for remove_char in [',', '$', '%', 'g', '.']:
                split_answer = split_answer.replace(remove_char, '')
            split_answer = split_answer.strip()
            if "true" in split_answer.lower():
                return "true"
            elif "false" in split_answer.lower():
                return "false"
            if "yes" in split_answer.lower():
                return "true"
            elif "no" in split_answer.lower():
                return "false"
            return split_answer.lower()
        else:
            split_answer = cot_answer.split('Answer:')[1].strip()
            for remove_char in [',', '$', '%', 'g']:
                split_answer = split_answer.replace(remove_char, '')
            pattern = r'\d+/\d+|\d+\.\d+|\d+'
            numbers_in_string = re.findall(pattern, split_answer)
            if len(numbers_in_string) >0:
                return numbers_in_string[0]
    elif allow_answer and "answer" in cot_answer:
        if args.dataset == "strategyqa":
            split_answer = cot_answer.split('answer')[1].strip()
            for remove_char in [',', '$', '%', 'g', '.']:
                split_answer = split_answer.replace(remove_char, '')
            split_answer = split_answer.strip()
            if "true" in split_answer.lower():
                return "true"
            elif "false" in split_answer.lower():
                return "false"
            if "yes" in split_answer.lower():
                return "true"
            elif "no" in split_answer.lower():
                return "false"
            return split_answer.lower()
        else:
            split_answer = cot_answer.split('answer')[1].strip()
            for remove_char in [',', '$', '%', 'g']:
                split_answer = split_answer.replace(remove_char, '')
            pattern = r'\d+/\d+|\d+\.\d+|\d+'
            numbers_in_string = re.findall(pattern, split_answer)
            if len(numbers_in_string) >0:
                return numbers_in_string[0]

    return None

def get_number(string):
    if re.match(r'.*\..*\.\d$', string) and string.endswith('.0'):
        string = string[:-2]
    # Updated regex pattern to handle negative numbers
    pattern = r'-?\d+/\d+|-?\d+\.\d+|-?\d+'
    numbers_in_string = re.findall(pattern, string)
    if len(numbers_in_string) > 0:
        num = numbers_in_string[0]
        if '/' in num:
            numerator, denominator = map(float, num.split('/'))
            number_in_string = numerator / denominator
        else:
            number_in_string = float(num)
        return number_in_string
    return None


def evaluate_answer(args, string, number_to_compare, tolerance=0.01):
    if string is None:
        return False
    if "gsm8k" in args.dataset or "svamp" in args.dataset or "gsm_hard" in args.dataset or "aime" in args.dataset:
        if isinstance(number_to_compare, str):
            number_to_compare = number_to_compare.strip()
            number_to_compare = number_to_compare.replace(',', '')
            number_to_compare = get_number(number_to_compare)
    elif "strategyqa" in args.dataset:
        if string.lower() == number_to_compare.lower():
            return True
        else:
            return False
    numbers_in_string = get_number(string)
    if numbers_in_string is not None and number_to_compare is not None:
        if abs(numbers_in_string - number_to_compare) <= tolerance:
            return True
    return False


def parse_llm_response(args, raw_responses_list):
    def instruction_following_score(input_text):
        score = 0
        if input_text.find('Answer:')!=-1:
            score += .5
        if not input_text.startswith('Answer'):
            score += .5
        return score
    
    if "google/gemma-2-2b" in args.model_name.lower():
        final_responses = []
        scores = []
        for response in raw_responses_list:
            print(response)
            start = response.find('<unused1>')
            end = response.find('<end_of_turn>')
            final_responses.append(response[start+9:end].replace('<eos>', '').strip('Questio'))
            print(final_responses)
            print('---\n')
            scores.append(instruction_following_score(final_responses[-1]))
    if "meta-llama/llama-3.1-8b" in args.model_name.lower():
        final_responses = []
        scores = []
        for response in raw_responses_list:
            try:
                temp = response.split('Let’s think step by step.')[-1].split('Question')[0].replace('<|end_of_text|>', '').strip()
            except:
                temp = response.strip()
            # print(temp)
            final_responses.append(temp)
            scores.append(instruction_following_score(final_responses[-1]))
    if args.model_name=="mistralai/Mistral-7B-Instruct-v0.1":
        final_responses = []
        scores = []
        for response in raw_responses_list:
            # print(response)
            try:
                temp = response.split('[control_760]')[1].replace('</s>', '').strip()
                if len(temp.split('Answer:'))>2:
                    temp = 'Answer:'.join(temp.split('Answer:')[:2])
            except:
                temp = response.replace('</s>', '').strip()
            final_responses.append(temp)
            scores.append(instruction_following_score(final_responses[-1]))
    return final_responses, scores


    
    # output_except_prompt = [item.split("Let’s think step by step.")[-1].replace('Question', '').split('\n\n')[0].strip() for item in output_except_prompt] # Gemma 
    # # output_except_prompt = [item.split("<|reserved_special_token_20|>")[-1].split('Question')[0].strip() for item in output_except_prompt] # LLama 
