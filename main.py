from data import get_dataset
from model import features,classifier,metrics
import tqdm
# from model import get_features,classifier

path="random_split/"

test,val,train =get_dataset.get_df(path)


# all_classes=features.get_classes(train)

# print(len(all_classes))

#Due to limitation of computing capacities, we will only consider the 1000 most important classes

classes=features.get_classes_top1000(train)

#Process inputs 
train_processed=features.process_dataset(train, classes)
test_processed=features.process_dataset(test, classes)
val_processed=features.process_dataset(val, classes)
print(train_processed.shape)
print(test_processed.shape)
print(val_processed.shape)
#Process labels
y_train=features.process_labels(train,classes)
y_test=features.process_labels(test,classes)
y_val=features.process_labels(val,classes)

print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

model=classifier.build_model()
print(model.summary())

history=classifier.train_model(model,train_processed,y_train,val_processed,y_val)

classifier.plot_accuracy_train_val(history)
