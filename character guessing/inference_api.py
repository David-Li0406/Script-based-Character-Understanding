import json
import openai
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
# enter your openai api key here
api_key = ''
openai.api_key=api_key
import random
random.seed(42)

choosen_sample = {"text": " P0 : hey, sorry about that. P1 : no, we're sorry. we never should have been comparing relationships in the first place. P2 : why? we won. you know, i say, next, we take on koothrappali and his dog. really give ourselves a challenge. P3 : i just want to say one more thing about this. just because penny and i are very different people does not mean that we're a bad couple. P2 : the answer is one simple test away. hmm? you know, it's like when i thought there was a possum in my closet. did i sit around wondering? no, i sent leonard in with a pointy stick and a bag. P3 : i killed his chewbacca slippers. P0 : let's just take the test. P3 : no, no, no, i don't want to. P0 : oh, well, 'cause you know we're gonna do bad. P3 : because it doesn't matter. i don't care if we're a ten or a two. P2 : or a one. a one is possible. P3 : marriage is scary. you're scared, i'm scared. but it doesn't make me not want to do it. it, it just makes me want to hold your hand and do it with you. P0 : leonard. P1 : it would make me so happy if you said things like that. P2 : we got an eight-point-two. trust me, you're happy.", "label": [2, 5, 0, 1], "showname": "The_Big_Bang_Theory", "summary": "Howard is asked to throw out the first pitch at a baseball game for the LA Angels of Anaheim, as an astronaut, to celebrate Space Day. Despite practicing, he cannot reach home plate. At the game Howard decides to use a prototype of the Mars rover to deliver the ball, but the Rover moves so slowly across the grass to deliver the ball that Howard is booed by the crowd and his friends in the stands. Earlier Sheldon and Amy are joined by Penny and Leonard on a double date at a pub, and are baffled when Sheldon says the Shamy relationship is the stronger of the two, because they scored an 8.2 out of 10 on a scientific test used to measure a couple's compatibility. Leonard wants to take the test, but Penny refuses, worried that she and Leonard have nothing in common even though they are engaged. Leonard admits he shares this fear, but says it only strengthens his resolve that they can face it together. Penny is overwhelmed by Leonard's commitment, while Amy wishes Sheldon would say romantic things like that to her. Sheldon tells her that she should trust him that she is already happy because of their test score."}

template = '''Here is an example: {}. 
Following this example and tell me {} below are which character from TV show "{}", please choose from {}, {}, {}, {}, {} and {}: {}. Prediction results:'''
# template_summary = 'Tell me {} below is which character from TV show "{}", you can refer to the summary of that episode: {} Please choose from {}, {}, {}, {}, {} and {}: {}'
demonstration_template = 'The following scripts is from TV show "{}": {}, Prediction results: {}'

def request_api(character_id, showname, scripts_content, demonstration, *all_character, summary=None):
    try:
        if summary is None:
            content = template.format(demonstration, character_id, showname, *all_character, scripts_content)
        else:
            content = template_summary.format(character_id, showname, summary, *all_character, scripts_content)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301", 
            messages=[{"role": "user", "content": content}]
        )
        ret = completion["choices"][0]["message"]["content"].strip()
        return ret
    except Exception as E:
        time.sleep(2)
        return request_api(character_id, showname, scripts_content, demonstration, *all_character, summary=summary)

def build_demonstration(sample_demonstration, reversed_label_dict):
    # print(sample_demonstration)
    ans = ''
    for ids, label in enumerate(sample_demonstration['label']):
        character_name = reversed_label_dict[sample_demonstration['showname']][label]
        ans += 'P{} is {}, '.format(ids, character_name)
    demonstration =  demonstration_template.format(sample_demonstration['showname'], sample_demonstration['text'], ans)
    return demonstration

def evaluate_all_in_one():
    data = json.load(open('dataset/API/test.json', 'r'))
    train_data = json.load(open('dataset/API/train.json', 'r'))
    train_data_dict = {}
    for item in train_data:
        if item['showname'] not in train_data_dict.keys():
            train_data_dict[item['showname']] = []
        train_data_dict[item['showname']].append(item)
    label_dict = json.load(open('dataset/API/label_dict.json', 'r'))
    reversed_label_dict = {showname:{label: character for character, label in label_dict[showname].items()} for showname in label_dict.keys()}
    all_character_dict = {k:tuple(v.keys()) for k, v in label_dict.items()}
    showname_id_dict = {k:i for i, k in enumerate(label_dict)}

    true_labels = []
    pred_labels = []
    micro_f1 = []

    for data_ids, ins in tqdm(enumerate(data)):
        text, label, showname, summary = ins['text'], ins['label'], ins['showname'], ins['summary']
        data[data_ids]['pred'] = {}
        data[data_ids]['pred_summary'] = {}
        character_ids = ['P'+str(i) for i in range(len(label))]
        character_ids = ', '.join(character_ids)
        all_character = all_character_dict[showname]
        sample_demonstration = random.sample(train_data_dict[ins['showname']],1)[0]
        demonstration = build_demonstration(sample_demonstration, reversed_label_dict)
        pred_text = request_api(character_ids, showname, text, demonstration, *all_character)
        character_pos = {}
        for character in all_character:
            pos = pred_text.lower().find(character.lower())
            if pos != -1:
                character_pos[pos] = character
        if len(character_pos) != len(label):
            print('character number in the response do not match with the input.')
            continue

        character_pos = sorted(character_pos.items(), key=lambda x: x[0])
        pred = [label_dict[showname][character[1]] for character in character_pos]
        micro_f1.extend([1 if p == l else 0 for p, l in zip(pred, label)])
        true_labels.extend([6*showname_id_dict[showname]+l for l in label])
        pred_labels.extend([6*showname_id_dict[showname]+p for p in pred])

        f1_micro = sum(micro_f1)/len(micro_f1)
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        print('micro f1 for all in one: ', f1_micro)
        print('macro f1 for all in one: ', f1_macro)



    f1_micro = sum(micro_f1)/len(micro_f1)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    print('micro f1 for all in one: ', f1_micro)
    print('macro f1 for all in one: ', f1_macro)
        
def test():
    data = json.load(open('dataset/API/test.json', 'r'))
    train_data = json.load(open('dataset/API/train.json', 'r'))
    train_data_dict = {}
    for item in train_data:
        if item['showname'] not in train_data_dict.keys():
            train_data_dict[item['showname']] = []
        train_data_dict[item['showname']].append(item)
    label_dict = json.load(open('dataset/API/label_dict.json', 'r'))
    reversed_label_dict = {showname:{label: character for character, label in label_dict[showname].items()} for showname in label_dict.keys()}
    all_character_dict = {k:tuple(v.keys()) for k, v in label_dict.items()}
    showname_id_dict = {k:i for i, k in enumerate(label_dict)}

    text, label, showname, summary = choosen_sample['text'], choosen_sample['label'], choosen_sample['showname'], choosen_sample['summary']
    # data[data_ids]['pred'] = {}
    # data[data_ids]['pred_summary'] = {}
    character_ids = ['P'+str(i) for i in range(len(label))]
    character_ids = ', '.join(character_ids)
    all_character = all_character_dict[showname]
    sample_demonstration = random.sample(train_data_dict[choosen_sample['showname']],1)[0]
    demonstration = build_demonstration(sample_demonstration, reversed_label_dict)
    pred_text = request_api(character_ids, showname, text, demonstration, *all_character)
    print(pred_text)

def main():
    data = json.load(open('dataset/API/test.json', 'r'))
    label_dict = json.load(open('dataset/API/label_dict.json', 'r'))
    reversed_label_dict = {showname:{label: character for character, label in label_dict[showname].items()} for showname in label_dict.keys()}
    all_character_dict = {k:tuple(v.keys()) for k, v in label_dict.items()}
    showname_id_dict = {k:i for i, k in enumerate(label_dict)}
    pred_text = request_api(character_id, showname, text, *all_character)

    true_labels = []
    pred_labels = []
    micro_f1 = []
    # add summary
    true_labels_summary = []
    pred_labels_summary = []
    micro_f1_summary = []
    for data_ids, ins in tqdm(enumerate(data)):
        text, label, showname, summary = ins['text'], ins['label'], ins['showname'], ins['summary']
        data[data_ids]['pred'] = {}
        data[data_ids]['pred_summary'] = {}
        for i in range(len(label)):
            character_id = 'P{}'.format(str(i))
            all_character = all_character_dict[showname]
            pred_text = request_api(character_id, showname, text, *all_character)
            pred_text_summary = request_api(character_id, showname, text, *all_character, summary=summary)
            true_text = reversed_label_dict[showname][label[i]]
            data[data_ids]['pred'][character_id] = pred_text
            data[data_ids]['pred_summary'][character_id] = pred_text_summary
            
            def calculate_f1(micro, true, pred, text):
                invalid_character_ids = ['P'+str(idx) for idx in range(len(label)) if idx != i]
                if any([False if character_ids not in text else True for character_ids in invalid_character_ids]):
                    return

                micro.append(1 if true_text.lower() in text.lower() else 0)
                find = False
                for character_name in all_character_dict[showname]:
                    if character_name.lower() in text.lower():
                        pred.append(6*showname_id_dict[showname]+label_dict[showname][character_name])
                        find = True
                        break
                if not find:
                    return
                true.append(6*showname_id_dict[showname]+label[i])

            # for summary
            calculate_f1(micro_f1_summary, true_labels_summary, pred_labels_summary, pred_text_summary)

            # without summary
            calculate_f1(micro_f1, true_labels, pred_labels, pred_text)

    # for summary
    # f1_micro_summary = sum(micro_f1_summary)/len(micro_f1_summary)
    # f1_macro_summary = f1_score(true_labels_summary, pred_labels_summary, average='macro')
    # print('micro f1 with summary: ', f1_micro_summary)
    # print('macro f1 with summary: ', f1_macro_summary)

    # without summary
    f1_micro = sum(micro_f1)/len(micro_f1)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    print('micro f1 without summary: ', f1_micro)
    print('macro f1 without summary: ', f1_macro)

    with open('dataset/API/test_result_ChatGPT.json', 'w') as fi:
        fi.write(json.dumps(data))

if __name__ == '__main__':
    evaluate_all_in_one()
    # test()