from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import time
import datetime
from pathlib import Path

from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import BertForPreTraining, BertConfig, BertForMaskedLM

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from Model import JointLearningMemoryModel, dataloader
import os
import sys
sys.path.append(os.path.abspath("/home/yons/CharacterRelationMining/coref_model"))

from evaluators import Ceaf4_score, Bcube_score, MUC_score, links2clusters_new, score2clusters_new, Blanc_score, links2clusters, links2clusters, score2clusters
from transformers import *
import logging
# import apex


import datetime
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file_name = './logs/log-'+now_time
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    filename = log_file_name,
                    filemode = 'w',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def softmax_result_to_cluster(prev_links, mentions, prev_ids):
    # prev_ids include the dummy mentions, so need to
    # print(len(prev_ids))
    # print(len(mentions))
    for position, second_mention in enumerate(mentions):
        if prev_ids[position] == -1:
            continue
        first_mention = mentions[prev_ids[position]]
        prev_links[(first_mention, second_mention)] = 1


    return prev_links



def main():
    parser = ArgumentParser()

    parser.add_argument('--trn_data', type=Path, required=True)
    parser.add_argument('--dev_data', type=Path, required=True)
    parser.add_argument('--tst_data', type=Path, required=True)

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--model",
                        required=True,
                        type=str,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_epochs",
                        default=10,
                        type=int,
                        help="Total number of epochs for training.")

    parser.add_argument("--num_memory_layers",
                        default=2,
                        type=int,
                        help="Total number of layers in memory network .")

    parser.add_argument("--coref_weight",
                        default=1,
                        type=float,
                        help="The weight in the front of coref loss.")
    
    parser.add_argument("--linking_weight",
                        default=1,
                        type=float,
                        help="The weight in the front of linking loss.")
    
    parser.add_argument("--recap_weight",
                        default=0,
                        type=float,)

    parser.add_argument("--in_batch_weight",
                        default=0,
                        type=float,)

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="# Gradient clipping is not in AdamW anymore (so you can use amp without issue)")

    parser.add_argument('--output_examples',
                        action='store_true',
                        help="output some examples during training to do case study")
    
    parser.add_argument('--output_dir',
                        type=str,
                        default=None)

    parser.add_argument('--from_pretrained',
                        type=str,
                        default=None)

    parser.add_argument('--dev_keys', type=Path, required=True)
    parser.add_argument('--trn_keys', type=Path, required=True)
    parser.add_argument('--tst_keys', type=Path, required=True)

    parser.add_argument('--test', action='store_true', help='To do evaluation on the test set')

    parser.add_argument('--train', action='store_true', help='To do training')

    parser.add_argument('--evaluate_interval', default=1, type=int)

    args = parser.parse_args()
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {},  16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model = JointLearningMemoryModel(args.model, args.num_memory_layers, args.coref_weight, args.linking_weight, args.in_batch_weight, args.recap_weight)
    if args.from_pretrained != None:
        print("Loading model from {}".format(args.from_pretrained))
        model.load_state_dict(torch.load(args.from_pretrained))

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False, eps=1e-5)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level="O0")



    if args.model[:8] == "SpanBERT":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model)

    train_dataloader = dataloader(args.trn_data, tokenizer)
    dev_dataloader = dataloader(args.dev_data, tokenizer)
    test_dataloader = dataloader(args.tst_data, tokenizer)

    num_training_steps = train_dataloader.length * args.num_epochs
    num_warmup_steps = args.warmup_proportion * num_training_steps

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)



    with open(args.trn_keys, "r") as fin:
        train_triplet = json.load(fin)
        gold_train_cluster = links2clusters(train_triplet)

    with open(args.dev_keys, "r") as fin:
        dev_triplet = json.load(fin)
        dev_triplet_dict = {(trip[0], trip[1]): trip[2] for trip in dev_triplet}
        gold_dev_cluster = links2clusters(dev_triplet)

    with open(args.tst_keys, "r") as fin:
        test_triplet = json.load(fin)
        test_triplet_dict = {(trip[0], trip[1]): trip[2] for trip in test_triplet}
        gold_test_cluster = links2clusters(test_triplet)

    if args.train:
        best_score = -1.0
        for epoch_counter in range(args.num_epochs):
            train_dataloader.reset()
            dev_dataloader.reset()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            tr_contrastive_loss = 0
            nb_con_examples = 0

            logger.info("***** Running training *****")
            logger.info("Epoch: " + str(epoch_counter))
            # First training
            model.train()
            with tqdm(total=train_dataloader.length, desc=f"Trn Epoch {epoch_counter}") as pbar:
                for _ in range(train_dataloader.length):
                    input_values, mention_ids = train_dataloader.get_document()
                    batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids, recap_ids, recap_mention_start, recap_mention_end, recap_character_ids = input_values
                    if mention_start.shape[0] > 0:
                        try:
                            contrastive_loss, contrastive_num = None, None
                            mean_con = 0.0
                            if args.recap_weight != 0.0 or args.in_batch_weight != 0.0:
                                loss, contrastive_loss, contrastive_num = model(batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids, recap_ids, recap_mention_start, recap_mention_end, recap_character_ids)
                                loss = loss + contrastive_loss
                            else:
                                loss, _, _ = model(batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids)
                            if args.fp16:
                                with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss.backward()

                            tr_loss += loss.item()
                            nb_tr_examples += mention_start.size(0)
                            nb_tr_steps += 1

                            if contrastive_loss is not None:
                                tr_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
                                nb_con_examples += contrastive_num

                            if nb_con_examples != 0:
                                mean_con = tr_contrastive_loss / nb_con_examples

                            mean_loss = tr_loss / nb_tr_examples

                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            scheduler.step()

                            optimizer.zero_grad()
                        except Exception as E:
                            print(E)
                            print("cannot fit in:", _)
                            if args.recap_weight !=0 or args.in_batch_weight != 0:
                                pbar.set_postfix_str(f"Contrastive Loss: {mean_con:.5f}")
                            else:
                                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                            pbar.update(1)
                            continue


                    if args.recap_weight !=0 or args.in_batch_weight != 0:
                        pbar.set_postfix_str(f"Contrastive Loss: {mean_con:.5f}")
                    else:
                        pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    pbar.update(1)
            del loss


            model.eval()

            if epoch_counter % args.evaluate_interval == 0 :
                logger.info("***** Running evaluation on dev *****")
                with tqdm(total=dev_dataloader.length, desc=f"Trn Epoch {epoch_counter}") as pbar2:

                    all_prev_link_dict = {t: 0 for t in dev_triplet_dict.keys()}
                    all_outputs = []
                    all_labels = []


                    for _ in range(dev_dataloader.length):
                        input_values, mention_ids = dev_dataloader.get_document()
                        batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids, recap_ids, recap_mention_start, recap_mention_end, recap_character_ids = input_values

                        if mention_start.shape[0] > 0:
                            link_result, linking_result = model(batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids)

                            prev_positions = [i - 1 for i in link_result.cpu().detach().numpy()]
                            prev_positions = prev_positions[1:]

                            all_outputs.extend(linking_result.cpu().detach().numpy().tolist())
                            all_labels.extend(character_ids.cpu().numpy().tolist())

                            # document_clusters = softmax_result_to_cluster(mention_ids, prev_positions)
                            all_prev_link_dict =  softmax_result_to_cluster (all_prev_link_dict, mention_ids, prev_positions)

                        pbar2.update(1)

                    triplet_scores = [[t[0], t[1], score] for t, score in all_prev_link_dict.items()]

                    all_clusters = links2clusters(triplet_scores)

                    linking_accuracy = accuracy_score(all_labels, all_outputs)
                    linking_macro_pr = precision_score(all_labels, all_outputs, average='macro')
                    linking_micro_pr = precision_score(all_labels, all_outputs, average="micro")
                    linking_macro_rc = recall_score(all_labels, all_outputs, average='macro')
                    linking_micro_rc = recall_score(all_labels, all_outputs, average="micro")
                    linking_macro_f1 = f1_score(all_labels, all_outputs, average='macro')
                    linking_micro_f1 = f1_score(all_labels, all_outputs, average="micro")
                    linking_class_f1 = f1_score(all_labels, all_outputs, average=None)
                    bcube_score = Bcube_score(gold_dev_cluster, all_clusters)
                    ceaf4_score = Ceaf4_score(gold_dev_cluster, all_clusters)
                    blanc_score = Blanc_score(gold_dev_cluster, all_clusters)
                    
                    logger.info("Linking accuracy: " + str(linking_accuracy))
                    logger.info("Linking macro Pr: " + str(linking_macro_pr))
                    logger.info("Linking micro Pr: " + str(linking_micro_pr))
                    logger.info("Linking macro Rc: " + str(linking_macro_rc))
                    logger.info("Linking micro Rc: " + str(linking_micro_rc))
                    logger.info("Linking macro F1: " + str(linking_macro_f1))
                    logger.info("Linking micro F1: " + str(linking_micro_f1))

                    logger.info("Linking class F1: " + str(linking_class_f1))

                    # logger.info("MUC___score: " + str(MUC_score(gold_test_cluster, all_clusters)))
                    logger.info("Bcube_score: " + str(bcube_score))
                    logger.info("Ceaf4_score: " + str(ceaf4_score))
                    logger.info("Blanc_score: " + str(blanc_score))
                    logger.info(f"Loss: {mean_loss:.5f}")

                    linking_score = linking_macro_f1 + linking_micro_f1
                    coreference_score = bcube_score[2] + ceaf4_score[2] + blanc_score[2]
                    score = args.linking_weight * linking_score + args.coref_weight * coreference_score
                    if score > best_score:
                        best_score = score
                        print('[*] Saving the checkpoint with best dev score at epoch {}'.format(epoch_counter))
                        if not os.path.exists(args.output_dir):
                            os.mkdir(args.output_dir)
                        if os.path.exists(os.path.join(args.output_dir,'pytorch_model.pt')):
                            os.remove(os.path.join(args.output_dir,'pytorch_model.pt'))
                        torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model.pt'))


                del link_result, linking_result

                if args.output_examples:
                    with open("./logs/dev_samples.json", "w") as fout:
                        logger.info("***** Saving Some Dev Results *****")
                        with tqdm(total=dev_dataloader.length, desc=f"Dev Epoch {epoch_counter}") as pbar3:

                            for _ in range(dev_dataloader.length):
                                input_values, mention_ids = dev_dataloader.get_document()
                                batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids, recap_ids, recap_mention_start, recap_mention_end, recap_character_ids = input_values

                                if mention_start.shape[0] > 0:
                                    link_result, linking_result = model(batch_ids, masks, mention_seg, mention_start, mention_end,
                                                                        speaker_ids)

                                    coref_prev_positions = [int(i - 1) for i in link_result.cpu().detach().numpy()]
                                    coref_prev_positions = coref_prev_positions[1:]

                                    linking_output_character_id = linking_result.cpu().detach().numpy().tolist()
                                    linking_output_character_label = character_ids.cpu().numpy().tolist()

                                    linking_output_character_id = [int(i) for i in linking_output_character_id]
                                    linking_output_character_label = [int(i) for i in linking_output_character_label]

                                    dev_result = {
                                        "mention_ids": mention_ids,
                                        "coref_prev_positions":coref_prev_positions,
                                        "linking_output_character_id": linking_output_character_id
                                    }

                                    fout.write(json.dumps(dev_result) + "\n")
                                else:
                                    dev_result = {
                                        "mention_ids": [],
                                        "coref_prev_positions": [],
                                        "linking_output_character_id": []
                                    }
                                    fout.write(json.dumps(dev_result) + "\n")


                                pbar3.update(1)



                            del link_result, linking_result

    if args.test:

        with open('../data/cluster_scene_tst.json', 'r') as fin:
            gold_test_cluster_scene = json.load(fin)
        gold_test_cluster_scenes = {k: list([set(item) for item in v.values()]) for k, v in gold_test_cluster_scene.items()}
        mention_set_scene = []
        for clusters in gold_test_cluster_scenes.values():
            mention_set_scene.append(set())
            for cluster in clusters:
                for mention in cluster:
                    mention_set_scene[-1].add(mention)

        test_dataloader.reset()
        model.eval()
        logger.info("***** Running evaluation on test *****")
        with tqdm(total=test_dataloader.length, desc="Testing") as pbar2:

            all_prev_link_dict = {t: 0 for t in test_triplet_dict.keys()}
            all_outputs = []
            all_labels = []

            all_outputs_scene = []
            all_labels_scene = []

            test_samples = {}

            for _, (scene_id, gold_test_cluster_scene), mention_set in zip(range(test_dataloader.length), gold_test_cluster_scenes.items(), mention_set_scene):
                input_values, mention_ids = test_dataloader.get_document()
                batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids, recap_ids, recap_mention_start, recap_mention_end, recap_character_ids = input_values

                if mention_start.shape[0] > 0:
                    with torch.no_grad():
                        link_result, linking_result = model(batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids)

                    prev_positions = [i - 1 for i in link_result.cpu().detach().numpy()]
                    prev_positions = prev_positions[1:]

                    all_outputs.extend(linking_result.cpu().detach().numpy().tolist())
                    all_labels.extend(character_ids.cpu().numpy().tolist())

                    all_outputs_scene.append(linking_result.cpu().detach().numpy().tolist())
                    all_labels_scene.append(character_ids.cpu().numpy().tolist())

                    # document_clusters = softmax_result_to_cluster(mention_ids, prev_positions)
                    all_prev_link_dict =  softmax_result_to_cluster (all_prev_link_dict, mention_ids, prev_positions)

                    all_prev_link_dict_scene = {t: 0 for t in test_triplet_dict.keys() if t[0] in mention_set and t[1] in mention_set}
                    all_prev_link_dict_scene = softmax_result_to_cluster(all_prev_link_dict_scene, mention_ids, prev_positions)
                    triplet_scores_scene = [[t[0], t[1], score] for t, score in all_prev_link_dict_scene.items()]
                    all_clusters_scene = links2clusters(triplet_scores_scene)
                    linking_macro_f1 = f1_score(all_labels_scene[-1], all_outputs_scene[-1], average='macro')
                    linking_micro_f1 = f1_score(all_labels_scene[-1], all_outputs_scene[-1], average="micro")
                    bcube_score_f1 = Bcube_score(gold_test_cluster_scene, all_clusters_scene)[2]
                    ceaf4_score_f1 = Ceaf4_score(gold_test_cluster_scene, all_clusters_scene)[2]
                    blanc_score_f1 = Blanc_score(gold_test_cluster_scene, all_clusters_scene)[2]

                    test_samples[scene_id] = {
                        'all_outputs_scene': all_outputs_scene[-1],
                        'all_labels_scene': all_labels_scene[-1],
                        'all_clusters_scene': [list(item) for item in all_clusters_scene],
                        'gold_test_cluster_scene': [list(item) for item in gold_test_cluster_scene],
                        'linking_macro_f1': linking_macro_f1,
                        'linking_micro_f1': linking_micro_f1,
                        'bcube_score_f1': bcube_score_f1,
                        'ceaf4_score_f1': ceaf4_score_f1,
                        'blanc_score_f1': blanc_score_f1,
                    }
                    


                pbar2.update(1)

            with open('./logs/test_samples_ours.json', "w") as fout:
                fout.write(json.dumps(test_samples))

            triplet_scores = [[t[0], t[1], score] for t, score in all_prev_link_dict.items()]

            all_clusters = links2clusters(triplet_scores)

            linking_accuracy = accuracy_score(all_labels, all_outputs)
            linking_macro_pr = precision_score(all_labels, all_outputs, average='macro')
            linking_micro_pr = precision_score(all_labels, all_outputs, average="micro")
            linking_macro_rc = recall_score(all_labels, all_outputs, average='macro')
            linking_micro_rc = recall_score(all_labels, all_outputs, average="micro")
            linking_macro_f1 = f1_score(all_labels, all_outputs, average='macro')
            linking_micro_f1 = f1_score(all_labels, all_outputs, average="micro")
            linking_class_f1 = f1_score(all_labels, all_outputs, average=None)
            bcube_score = Bcube_score(gold_test_cluster, all_clusters)
            ceaf4_score = Ceaf4_score(gold_test_cluster, all_clusters)
            blanc_score = Blanc_score(gold_test_cluster, all_clusters)
            
            logger.info("Linking accuracy: " + str(linking_accuracy))
            logger.info("Linking macro Pr: " + str(linking_macro_pr))
            logger.info("Linking micro Pr: " + str(linking_micro_pr))
            logger.info("Linking macro Rc: " + str(linking_macro_rc))
            logger.info("Linking micro Rc: " + str(linking_micro_rc))
            logger.info("Linking macro F1: " + str(linking_macro_f1))
            logger.info("Linking micro F1: " + str(linking_micro_f1))

            logger.info("Linking class F1: " + str(linking_class_f1))

            # logger.info("MUC___score: " + str(MUC_score(gold_test_cluster, all_clusters)))
            logger.info("Bcube_score: " + str(bcube_score))
            logger.info("Ceaf4_score: " + str(ceaf4_score))
            logger.info("Blanc_score: " + str(blanc_score))

            del link_result, linking_result


        # output test examples
        # with open("./logs/test_samples.json", "w") as fout:
        #     logger.info("***** Saving Some Test Results *****")
        #     with tqdm(total=test_dataloader.length) as pbar3:

        #         for _ in range(test_dataloader.length):
        #             input_values, mention_ids = test_dataloader.get_document()
        #             batch_ids, masks, mention_seg, mention_start, mention_end, speaker_ids, link_start, link_end, character_ids, recap_ids, recap_mention_start, recap_mention_end, recap_character_ids = input_values

        #             if mention_start.shape[0] > 0:
        #                 with torch.no_grad():
        #                     link_result, linking_result = model(batch_ids, masks, mention_seg, mention_start, mention_end,
        #                                                     speaker_ids)

        #                 coref_prev_positions = [int(i - 1) for i in link_result.cpu().detach().numpy()]
        #                 coref_prev_positions = coref_prev_positions[1:]

        #                 linking_output_character_id = linking_result.cpu().detach().numpy().tolist()
        #                 linking_output_character_label = character_ids.cpu().numpy().tolist()

        #                 linking_output_character_id = [int(i) for i in linking_output_character_id]
        #                 linking_output_character_label = [int(i) for i in linking_output_character_label]

        #                 test_result = {
        #                     "mention_ids": mention_ids,
        #                     "coref_prev_positions":coref_prev_positions,
        #                     "linking_output_character_id": linking_output_character_id
        #                 }

        #                 fout.write(json.dumps(test_result) + "\n")
        #             else:
        #                 test_result = {
        #                     "mention_ids": [],
        #                     "coref_prev_positions": [],
        #                     "linking_output_character_id": []
        #                 }
        #                 fout.write(json.dumps(test_result) + "\n")

        #             pbar3.update(1)






if __name__ == "__main__":
    main()
