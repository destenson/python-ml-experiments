from transformers import BertTokenizer, TFBertModel
from transformers import pipeline
import tensorflow as tf

transformers_models = [
    "audio-classification", # will return a [`AudioClassificationPipeline`].
    "automatic-speech-recognition", # will return a [`AutomaticSpeechRecognitionPipeline`].
    "depth-estimation", # will return a [`DepthEstimationPipeline`].
    "document-question-answering", # will return a [`DocumentQuestionAnsweringPipeline`].
    "feature-extraction", # will return a [`FeatureExtractionPipeline`].
    "fill-mask", # will return a [`FillMaskPipeline`]:.
    "image-classification", # will return a [`ImageClassificationPipeline`].
    "image-feature-extraction", # will return an [`ImageFeatureExtractionPipeline`].
    "image-segmentation", # will return a [`ImageSegmentationPipeline`].
    "image-to-image", # will return a [`ImageToImagePipeline`].
    "image-to-text", # will return a [`ImageToTextPipeline`].
    "mask-generation", # will return a [`MaskGenerationPipeline`].
    "object-detection", # will return a [`ObjectDetectionPipeline`].
    "question-answering", # will return a [`QuestionAnsweringPipeline`].
    "summarization", # will return a [`SummarizationPipeline`].
    "table-question-answering", # will return a [`TableQuestionAnsweringPipeline`].
    "text2text-generation", # will return a [`Text2TextGenerationPipeline`].
    "text-classification", #(alias `"sentiment-analysis"` available): will return a [`TextClassificationPipeline`].
    "text-generation", # will return a [`TextGenerationPipeline`]:.
    "text-to-audio", #(alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
    "token-classification", # (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
    "translation", # will return a [`TranslationPipeline`].
    "translation_xx_to_yy", # will return a [`TranslationPipeline`].
    "video-classification", # will return a [`VideoClassificationPipeline`].
    "visual-question-answering", # will return a [`VisualQuestionAnsweringPipeline`].
    "zero-shot-classification", # will return a [`ZeroShotClassificationPipeline`].
    "zero-shot-image-classification", # will return a [`ZeroShotImageClassificationPipeline`].
    "zero-shot-audio-classification", # will return a [`ZeroShotAudioClassificationPipeline`].
    "zero-shot-object-detection", # will return a [`ZeroShotObjectDetectionPipeline`].
]

def test_unmasker():
    unmasker = pipeline('fill-mask',
                        model='distilbert/distilroberta-base',
                        tokenizer='distilbert/distilroberta-base',
                        framework='pt',
                        # framework='tf',
                        )
    result = unmasker("Artificial Intelligence <mask> take over the world.")
    print(f"BERT {len(result)} pipeline results:", [r['token_str'] for r in result])

def test_image_to_text(image):
    i2t = pipeline('image-to-text',
                model='ydshieh/vit-gpt2-coco-en',
                framework='tf',
                )
    result = i2t(image)
    print(f"BERT {len(result)} pipeline results:", result)


def test_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertModel.from_pretrained('bert-base-cased')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    outputs = model(inputs)

    last_hidden_states = outputs.last_hidden_state,

    print(f"BERT last_hidden_states shape: {last_hidden_states[0].shape}")
    print(f"BERT last_hidden_states: {last_hidden_states[0]}")

