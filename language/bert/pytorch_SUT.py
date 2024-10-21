import array
import json
import os
import sys
import torch
import transformers
import mlperf_loadgen as lg
import numpy as np
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL
import torch.quantization

class BERT_PyTorch_SUT():
    def __init__(self, args):
        print("Loading BERT configs...")
        with open("bert_config.json") as f:
            config_json = json.load(f)

        config = BertConfig(
            attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
            hidden_act=config_json["hidden_act"],
            hidden_dropout_prob=config_json["hidden_dropout_prob"],
            hidden_size=config_json["hidden_size"],
            initializer_range=config_json["initializer_range"],
            intermediate_size=config_json["intermediate_size"],
            max_position_embeddings=config_json["max_position_embeddings"],
            num_attention_heads=config_json["num_attention_heads"],
            num_hidden_layers=config_json["num_hidden_layers"],
            type_vocab_size=config_json["type_vocab_size"],
            vocab_size=config_json["vocab_size"])

        self.network = args.network
        self.dev = torch.device("cpu")  # 使用CPU进行推理
        self.version = transformers.__version__

        print("Loading PyTorch model...")
        self.model = BertForQuestionAnswering(config)
        self.model.to(self.dev)
        self.model.eval()
        model_file = os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
        self.model.load_state_dict(torch.load(model_file, map_location=self.dev), strict=False)

        # 量化模型
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        batch_size = 64  # 设置批量大小为64

        for i in range(0, len(query_samples), batch_size):
            batch_samples = query_samples[i:i + batch_size]
            eval_features = [self.qsl.get_features(sample.index) for sample in batch_samples]
            self.process_sample(eval_features, [sample.id for sample in batch_samples])

    def process_sample(self, sample_inputs, query_ids=None):
        input_ids = []
        input_masks = []
        segment_ids = []

        for sample_input in sample_inputs:
            if self.network == "sut":
                input_ids.append(sample_input['input_ids'])
                input_masks.append(sample_input['input_mask'])
                segment_ids.append(sample_input['segment_ids'])
            else:
                input_ids.append(sample_input.input_ids)
                input_masks.append(sample_input.input_mask)
                segment_ids.append(sample_input.segment_ids)

        input_ids = torch.LongTensor(input_ids).to(self.dev)
        input_masks = torch.LongTensor(input_masks).to(self.dev)
        segment_ids = torch.LongTensor(segment_ids).to(self.dev)

        with torch.no_grad():
            model_output = self.model.forward(
                input_ids=input_ids,
                attention_mask=input_masks,
                token_type_ids=segment_ids
            )
            if self.version >= '4.0.0':
                start_scores = model_output.start_logits
                end_scores = model_output.end_logits
            else:
                start_scores, end_scores = model_output

            outputs = torch.stack([start_scores, end_scores], axis=-1).cpu().numpy()

            if self.network == "sut":
                return [output.tolist() for output in outputs]

            for i, output in enumerate(outputs):
                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_ids[i], bi[0], bi[1])
                lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_pytorch_sut(args):
    torch.set_num_threads(16)  # 设置使用的线程数为16
    return BERT_PyTorch_SUT(args)
