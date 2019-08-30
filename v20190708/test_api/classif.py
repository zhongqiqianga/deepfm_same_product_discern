from grpc.beta import implementations
import tensorflow as tf
import time
import grpc
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc



if __name__=="__main__":
    channel = grpc.insecure_channel('192.168.3.101:8500')
    # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # initialize a request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'deepfm4'
    request.model_spec.signature_name = 'my_signature'
    feat_index=[[19946, 36282, 36348, 36417, 36486, 36560, 36743, 38833, 41632, 43828, 46572, 49172, 49241, 49313, 49404, 49455, 49581, 51973, 54888, 57073, 59774, 62251, 62318, 62390, 62485, 62531, 62954, 65159, 67726, 70612, 73067, 75839, 115615, 119856, 119901, 119902, 119903, 119904, 119905, 119906, 119907, 119908, 119909, 119910, 119911, 119912, 119913, 119914, 119915, 119916, 119917, 121207, 130086, 136455], [8390, 36280, 36351, 36415, 36503, 36569, 36957, 39364, 41251, 43852, 46662, 49173, 49298, 49327, 49385, 49455, 50579, 52991, 54385, 56992, 59628, 62252, 62348, 62405, 62462, 62531, 63712, 66186, 67563, 70218, 72846, 88230, 115280, 119828, 119901, 119902, 119903, 119904, 119905, 119906, 119907, 119908, 119909, 119910, 119911, 119912, 119913, 119914, 119915, 119916, 119918, 119947, 129950, 136264]]
    feat_value=[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 374.29, 40.0, 54.0, 63.0, 1.0, 2.0, 0.0, 0.0, 452.22, 27.5, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 4696.99, 34.0, 36.0, 53.0, 2.0, 2.0, 0.0, 0.0, 4735.32, 27.96, 1.0, 1.0, 1.0, 1.0]]
    label=[[1.0]]
    train_phase=False
    dropout_keep_fm=[1.0,1.0]
    dropout_keep_deep=[1.0,1.0,1.0]
    request.inputs['feat_value'].CopyFrom(tf.contrib.util.make_tensor_proto(feat_value))
    request.inputs['feat_index'].CopyFrom(tf.contrib.util.make_tensor_proto(feat_index))
    request.inputs['label'].CopyFrom(tf.contrib.util.make_tensor_proto(label))
    request.inputs['train_phase'].CopyFrom(tf.contrib.util.make_tensor_proto(train_phase))
    request.inputs['dropout_keep_fm'].CopyFrom(tf.contrib.util.make_tensor_proto(dropout_keep_fm))
    request.inputs['dropout_keep_deep'].CopyFrom(tf.contrib.util.make_tensor_proto(dropout_keep_deep))
    start_time = time.time()
    result = stub.Predict(request)
    print(result)