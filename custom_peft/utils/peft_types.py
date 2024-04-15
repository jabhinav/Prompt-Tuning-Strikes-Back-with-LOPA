# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum


class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"


class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    MULTI_CAUSAL_LM = "MULTI_CAUSAL_LM"  # Custom Added
    CVAE_CAUSAL_LM = "CVAE_CAUSAL_LM"  # Custom Added
    IDPG_CAUSAL_LM = "IDPG_CAUSAL_LM"  # Custom Added
    CCVAE_CAUSAL_LM = "CCVAE_CAUSAL_LM"  # Custom Added
    LIB_CVAE_CAUSAL_LM = "LIB_CVAE_CAUSAL_LM"  # Custom Added
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
