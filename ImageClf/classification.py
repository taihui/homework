##### We will use the pretrained models on ImageNet to make predications
import pandas as pd
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image


def make_prediction(pretrained_model, input_image_file, label_file):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    input_image = Image.open(input_image_file)
    img_t = transform(input_image)
    batch_t = torch.unsqueeze(img_t, 0)

    out = pretrained_model(batch_t)

    with open(label_file) as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    #
    # print(classes[index[0]], percentage[index[0]].item())

    _, indices = torch.sort(out, descending=True)

    ### now, let's get its class labels and its confidence percentages
    top_5_predictions = []
    for idx in indices[0][:5]:
        cur_label = classes[idx]
        cur_conf =  round(percentage[idx].item(),3)
        cur_prediction = '({}: {}%)'.format(cur_label, cur_conf)
        top_5_predictions.append(cur_prediction)

    return top_5_predictions




if __name__ == '__main__':
    file_list = ['input/IMG_Cow.JPG', 'input/IMG_Bird_Crane.JPG']
    label_file = 'input/imagenet_classes.txt'


    for input_image_file in file_list:
        cur_image_name = input_image_file.split('/')[-1].replace('.JPG','')
        result_dict = {'Pretrained_Model':[], 'Top-5 Predictions (label: confidence)':[]}

        ###################################################################
        #------------------------ using AlexNet ------------------------#
        ###################################################################
        alexnet = models.alexnet(pretrained=True)
        alexnet.eval()
        top_5_predictions = make_prediction(alexnet, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('AlexNet')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')

        ###################################################################
        # ------------------------ using VGG19 ------------------------#
        ###################################################################
        vgg19 = models.vgg19_bn(pretrained=True)
        vgg19.eval()
        top_5_predictions = make_prediction(vgg19, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('VGG19')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')


        ###################################################################
        # ------------------------ using ResNet-50 ------------------------#
        ###################################################################
        resnet50 = models.resnet50(pretrained=True)
        resnet50.eval()
        top_5_predictions = make_prediction(resnet50, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('ResNet50')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')

        ###################################################################
        # ------------------------ using ResNet-152 ------------------------#
        ###################################################################
        resnet152 = models.resnet152(pretrained=True)
        resnet152.eval()
        top_5_predictions = make_prediction(resnet152, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('ResNet152')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')

        ###################################################################
        # ------------------------ using SqueezeNet ------------------------#
        ###################################################################
        squeezenet1_1 = models.squeezenet1_1(pretrained=True)
        squeezenet1_1.eval()
        top_5_predictions = make_prediction(squeezenet1_1, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('SqueezeNet')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')

        ###################################################################
        # ------------------------ using Densenet-121 -------------------#
        ###################################################################
        densenet121 = models.densenet121(pretrained=True)
        densenet121.eval()
        top_5_predictions = make_prediction(densenet121, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('Densenet121')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')

        ###################################################################
        # ------------------------ using Densenet-169 -------------------#
        ###################################################################
        densenet169 = models.densenet169(pretrained=True)
        densenet169.eval()
        top_5_predictions = make_prediction(densenet169, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('Densenet169')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')

        ###################################################################
        # ------------------------ using Inception-v3 -------------------#
        ###################################################################
        inception_v3 = models.inception_v3(pretrained=True)
        inception_v3.eval()
        top_5_predictions = make_prediction(inception_v3, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('Inception-v3')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')

        ###################################################################
        # ------------------------ using ResNeXt-101-32x8d -------------------#
        ###################################################################
        resnext101_32x8d = models.resnext101_32x8d(pretrained=True)
        resnext101_32x8d.eval()
        top_5_predictions = make_prediction(resnext101_32x8d, input_image_file, label_file)

        result_dict['Pretrained_Model'].append('ResNeXt-101-32x8d')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[0])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[1])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[2])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[3])

        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append(top_5_predictions[4])

        ### after finishing one model, let's add a blank-row
        result_dict['Pretrained_Model'].append('')
        result_dict['Top-5 Predictions (label: confidence)'].append('')


        ### after finishing all models, let's save the results
        result_df = pd.DataFrame.from_dict(result_dict)
        result_file = 'output/{}_predictions.csv'.format(cur_image_name)
        result_df.to_csv(result_file, index=False)

        print('Finished {}'.format(cur_image_name))