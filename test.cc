//これがないとエラーになる
#define COMPILER_MSVC

//#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"

//using namespace tensorflow;
//using namespace tensorflow::ops;

double targetFunction(double x1, double x2) {
    return 2.0 * x1 * x2 - 2.4 * x1 * x1 + 3.1 * x2 * x2 + 2.7 * x1 - 10.0 * x2 + 5.0;
}

int main() {
    int unit_num;
    std::cout << "ユニット数 : ";
    std::cin >> unit_num;
    float learning_rate;
    std::cout << "学習率 : ";
    std::cin >> learning_rate;
    int batch_size;
    std::cout << "バッチサイズ : ";
    std::cin >> batch_size;
    int epoch_num;
    std::cout << "エポック数 : ";
    std::cin >> epoch_num;

    Scope scope = Scope::NewRootScope();

    //入力,教師データのPlaceholder
    auto x = Placeholder(scope, DT_FLOAT);
    auto y = Placeholder(scope, DT_FLOAT);

    //中間層への重み、バイアス
    auto w1 = Variable(scope, { unit_num, 2 }, DT_FLOAT);
    auto assign_w1 = Assign(scope, w1, RandomNormal(scope, { unit_num, 2 }, DT_FLOAT));
    auto b1 = Variable(scope, { unit_num, 1 }, DT_FLOAT);
    auto assign_b1 = Assign(scope, b1, RandomNormal(scope, { unit_num, 1 }, DT_FLOAT));

    //出力層への重み、バイアス
    auto w2 = Variable(scope, { 1, unit_num  }, DT_FLOAT);
    auto assign_w2 = Assign(scope, w2, RandomNormal(scope, { 1, unit_num }, DT_FLOAT));
    auto b2 = Variable(scope, { 1, 1 }, DT_FLOAT);
    auto assign_b2 = Assign(scope, b2, RandomNormal(scope, { 1, 1 }, DT_FLOAT));

    //中間層
    auto hidden_layer = Relu(scope, Add(scope, MatMul(scope, w1, x), b1));

    //出力層
    auto output_layer = Add(scope, MatMul(scope, w2, hidden_layer), b2);

    //損失
    auto loss = ReduceMean(scope, Square(scope, Sub(scope, output_layer, y)), {0, 1});

    //勾配
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, { loss }, { w1, w2, b1, b2 }, &grad_outputs));

    //勾配降下を各変数に適用
    auto apply_w1 = ApplyGradientDescent(scope, w1, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[0] });
    auto apply_w2 = ApplyGradientDescent(scope, w2, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[1] });
    auto apply_b1 = ApplyGradientDescent(scope, b1, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[2] });
    auto apply_b2 = ApplyGradientDescent(scope, b2, Cast(scope, learning_rate, DT_FLOAT), { grad_outputs[3] });

    //入力、教師データのPlaceholderに流す実際のデータ
    Tensor x_data(DT_FLOAT, TensorShape{ 2, batch_size });
    Tensor y_data(DT_FLOAT, TensorShape{ 1, batch_size });

    //x_dataとy_dataにbatch_size分のランダムデータを格納する関数
    auto setData = [&x_data, &y_data](int batch_size) {
        static std::random_device seed_gen;
        static std::default_random_engine engine(seed_gen());
        static std::uniform_real_distribution<> dist(-10.0, 10.0);

        for (int i = 0; i < batch_size; i++) {
            double x1 = dist(engine), x2 = dist(engine);
            x_data.matrix<float>()(0, i) = (float)x1;
            x_data.matrix<float>()(1, i) = (float)x2;

            y_data.matrix<float>()(0, i) = (float)targetFunction(x1, x2);
        }
    };

    //出力を受け取るための変数
    std::vector<Tensor> outputs;

    //セッションを作成
    ClientSession session(scope);

    //重みとバイアスを初期化
    TF_CHECK_OK(session.Run({ assign_w1, assign_w2, assign_b1, assign_b2 }, nullptr));

    for (int e = 0; e <= epoch_num; e++) {
        //検証
        setData(batch_size);
        TF_CHECK_OK(session.Run({ { x, x_data }, { y, y_data } }, { loss }, &outputs));

        printf("epoch = %5d, loss = %12.1f\n", e, outputs[0].scalar<float>()());
        if (e == epoch_num) {
            break;
        }

        //学習
        setData(batch_size);
        TF_CHECK_OK(session.Run({ { x, x_data },{ y, y_data } }, { apply_w1, apply_b1, apply_w2, apply_b2 }, nullptr));
    }
}
