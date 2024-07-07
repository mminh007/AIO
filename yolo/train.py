import argparse
import os
from ultralytics import YOLOv10
# path dataset
#
def parse_args(input_args=None):

    parser = argparse.ArgumentParser(description="Traning script YOLOv10")
    parser.add_argument("--root_dir",
                        type=str,
    )
    parser.add_argument("--pretrained_model_name_or_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model",
    )
    parser.add_argument("--data_dir",
                        type=str,
                        required=True,
                        help="A folder containing the training data",
    )
    # parser.add_argument("--num_classes",
    #                     type=int,
    #                     default=None,
    # )
    parser.add_argument("--name_classes",
                        type=list,
    )
    parser.add_argument("--data_yaml",
                        type=str,
                        help="directory of yaml file containing data's infomation"
    )
    # config hyper parameters
    parser.add_argument("--input_size",
                        type=int,
    )
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=32,
                        help="Batch size (per device) for the training"
    )
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=50)
    parser.add_argument("--validation_check",
                        type=bool,
                        default=False
    )


    
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args

def create_data_config(path, num_classes, names):
    info = {
        "train": os.path.join(path, "train/images"),
        "val": os.path.join(path, "valid/images"),
        "test": os.path.join(path, "test/images"),
        "nc": num_classes,
        "names": names,
    }
    yaml_filepath = os.path.join(path, "data.yaml")
    with open(yaml_filepath, "w") as f:
        doc = yaml.dump(
            info,
            f,
            default_flow_style=None,
            sort_keys=False
        )


def main():
    args = parse_args()
    
    # check model 
    if args.pretrained_model_name_or_path is None:
        MODEL_PATH = os.path.join(args.root_dir, "models/")
        model = YOLOv10(MODEL_PATH)
    
    else:
        model = YOLOv10.from_pretrained(args.pretrained_model_name_or_path)
    
    
    # check file data.yaml
    if args.data_yaml is None:
        if os.path.exists(os.path.join(args.root_dir, "datasets/data.yaml")) == False:
            if args.name_classes is None:
                raise Exception("The --data_yaml or --name_classes is not None")
            
            else:
                import yaml
                datasets_path = os.path.join(args.root_dir, args.data_dir)
            
                create_data_config(path = datasets_path, num_classes = len(args.name_classes), 
                                   names= args.name_classes)
    
    model.train(data=os.path.join(args.root_dir, "datasets/data.yaml"),
                epochs = args.num_train_epochs,
                batch = args.train_batch_size,
                imgsz = args.input_size)
    
    if args.validation_check == True:
        checkpoint = "./runs/detect/train/weights/best.pt"
        model = YOLOv10(checkpoint)

        model.val(data=os.path.join(args.root_dir, "datasets/data.yaml"),
                epochs = args.num_train_epochs,
                batch = args.train_batch_size,
                imgsz = args.input_size)


if __name__ == "__main__":
    main()