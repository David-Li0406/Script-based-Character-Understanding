import json
import openai
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
from evaluators import Bcube_score, Blanc_score, Ceaf4_score
# enter your openai api key here
api_key = ''
openai.api_key=api_key

prompt = 'Here is an example: {}. Following this example and read the following conversation: "{}" The NO.{} "{}" in the utterance "{}" refer to which character? you should choose answer from {}'
demonstration_template = 'This is an conversation: {}, The NO.1 "{}" in the utterance "{}" refer to {}'

character_id_to_name = ["Ross Geller",
    "Rachel Green",
    "Chandler Bing",
    "Monica Geller",
    "Joey Tribbiani",
    "Phoebe Buffay",
    "Emily",
    "Richard Burke",
    "Carol Willick",
    "Ben Geller",
    "Peter Becker",
    "Judy Geller",
    "Barry Farber",
    "Jack Geller",
    "Kate Miller",
    "#OTHER#",
    "#GENERAL#"]

def request_api(utterance_text_scene, character_candidate, coreference_idx, coreference, utterance_text, demonstration):
    try:
        sys_prompt = prompt.format(demonstration, utterance_text_scene, coreference_idx, coreference, utterance_text, character_candidate)
        # print(sys_prompt)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301", 
            messages=[{"role": "user", "content": sys_prompt}]

        )
        ret = completion["choices"][0]["message"]["content"].strip()
        return ret
    except Exception as E:
        time.sleep(2)
        return request_api(utterance_text_scene, character_candidate, coreference_idx, coreference, utterance_text, demonstration)


def load_data_train():
    data = json.load(open('../data/character-identification-trn.json', 'r'))
    cost = 0

    utterance_text_all, coreference_character_all = [], []
    for episode in data['episodes']:
        for scene in episode['scenes']:
            utterance_text_scene = ''
            coreference_character_scene = []
            for utterance in scene['utterances']:
                utterance_text = utterance['speakers'][0] + ': '
                utterance_no_character = ''
                coreference_character = []
                for token, character_entity in zip(utterance['tokens'], utterance['character_entities']):
                    utterance_text += ' '.join(token)
                    utterance_no_character += ' '.join(token)
                    for entity in character_entity:
                        start, end = entity[0:2]
                        character = entity[2:]
                        coreference = ''.join(token[start:end])
                        coreference_character.append([coreference, character]) 

                coreference_character = [[utterance_no_character] + item for item in coreference_character]
                coreference_character_scene.append(coreference_character)
                utterance_text_scene += utterance_text + ' '
                cost += len(utterance_text_scene.split(' ')) * len(coreference_character)

            utterance_text_all.append(utterance_text_scene)
            coreference_character_all.append(coreference_character_scene)

    print('estimated cost:', cost/1000*0.002)

    return utterance_text_all, coreference_character_all

def load_data():
    data = json.load(open('../data/character-identification-tst.json', 'r'))
    cost = 0

    utterance_text_all, coreference_character_all = [], []
    for episode in data['episodes']:
        for scene in episode['scenes']:
            utterance_text_scene = ''
            coreference_character_scene = []
            for utterance in scene['utterances']:
                utterance_text = utterance['speakers'][0] + ': '
                utterance_no_character = ''
                coreference_character = []
                for token, character_entity in zip(utterance['tokens'], utterance['character_entities']):
                    utterance_text += ' '.join(token)
                    utterance_no_character += ' '.join(token)
                    for entity in character_entity:
                        start, end = entity[0:2]
                        character = entity[2:]
                        coreference = ''.join(token[start:end])
                        coreference_character.append([coreference, character]) 

                coreference_character = [[utterance_no_character] + item for item in coreference_character]
                coreference_character_scene.append(coreference_character)
                utterance_text_scene += utterance_text + ' '
                cost += len(utterance_text_scene.split(' ')) * len(coreference_character)

            utterance_text_all.append(utterance_text_scene)
            coreference_character_all.append(coreference_character_scene)

    print('estimated cost:', cost/1000*0.002)

    return utterance_text_all, coreference_character_all

def build_demonstration(utterance_text_all_train, coreference_character_all_train):
    import random
    random.seed(42)
    sample_idx = random.randint(0,len(utterance_text_all_train)-1)
    # print(coreference_character_all_train[sample_idx])
    for item in coreference_character_all_train[sample_idx]:
        if item == []:
            continue
        choosen = item[0]
        demonstration = demonstration_template.format(utterance_text_all_train[sample_idx], choosen[1], choosen[0], choosen[2][0])
        # print(utterance_text_all_train)
        return demonstration


def main():
    utterance_text_all, coreference_character_all = load_data()
    utterance_text_all_train, coreference_character_all_train = load_data_train()
    character_candidate = ', '.join(character_id_to_name)

    pred_labels, true_labels = [], []
    pred_labels_scene, true_labels_scene = [], []
    for utterance_text_scene, coreference_character_scene in tqdm(zip(utterance_text_all, coreference_character_all)):
        all_mention_scene = []
        pred_labels_scene.append([])
        true_labels_scene.append([])
        for coreference_character in coreference_character_scene:
            coreference_idx_dict = {}
            for utterance_text, coreference, character in coreference_character:
                all_mention_scene.append(coreference)
                if len(character) > 1:
                    continue
                if character[0] not in character_id_to_name:
                    continue
                if coreference not in coreference_idx_dict:
                    coreference_idx_dict[coreference] = 0
                coreference_idx_dict[coreference] += 1
                coreference_idx = coreference_idx_dict[coreference]
                demonstration = build_demonstration(utterance_text_all_train, coreference_character_all_train)
                # print(demonstration)
                # exit()
                pred_text = request_api(utterance_text_scene, character_candidate, coreference_idx, coreference, utterance_text, demonstration)
                # print('pred:', pred_text)
                # print('gt: ', character[0])

                true_label = character_id_to_name.index(character[0])
                true_labels.append(true_label)
                true_labels_scene[-1].append(true_label)
                find = False
                for character_name in character_id_to_name:
                    if character_name.lower() in pred_text.lower():
                        pred_labels.append(character_id_to_name.index(character_name))
                        pred_labels_scene[-1].append(character_id_to_name.index(character_name))
                        find = True
                        break
                if not find:
                    # OTHERS
                    pred_labels.append(15)
                    pred_labels_scene[-1].append(15)
        # break
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        f1_micro = f1_score(true_labels, pred_labels, average='micro')
        print('macro f1: ', f1_macro)
        print('micro f1: ', f1_micro)

    result = {
        'pred_labels': pred_labels_scene,
        'true_labels': true_labels_scene
    }
    # print(result)
    with open('api_result.json', 'w') as fi:
        fi.write(json.dumps(result))
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_micro = f1_score(true_labels, pred_labels, average='micro')
    print('macro f1: ', f1_macro)
    print('micro f1: ', f1_micro)
                

def coreference_resolution():
    result = json.load(open('api_result.json', 'r'))

    idx = 0
    pred_dict = {}
    true_dict = {}
    for pred_labels, true_labels in zip(result['pred_labels'], result['true_labels']):
        for pred, true in zip(pred_labels, true_labels):
            if pred not in pred_dict.keys():
                pred_dict[pred] = set()
            pred_dict[pred].add(idx)

            if true not in true_dict.keys():
                true_dict[true] = set()
            true_dict[true].add(idx)

            idx+=1

    pred_list = [v for v in pred_dict.values()]
    true_list = [v for v in true_dict.values()]
    print(Blanc_score(pred_list, true_list))
    print(Bcube_score(pred_list, true_list))
    print(Ceaf4_score(pred_list, true_list))



if __name__ == '__main__':
    main()
    coreference_resolution()