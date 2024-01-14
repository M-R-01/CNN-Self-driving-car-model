
from utils import *
from sklearn.model_selection import train_test_split


# Step1
path = 'myData'
data = importDataInfo(path)

# Step2
balanceData(data, display= False)

# Step3
imagesPath, steering = loadData(path, data)
#print(imagesPath[0], steering[0])

# Step4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steering,test_size=0.2,random_state=5)
print(len(xTrain))
print(len(xVal))

# Step5

# Step6

# Step7

# Step8
model = createModel()
model.summary()

# Step9
history = model.fit(batchCreate(xTrain,yTrain,50,1),steps_per_epoch=150,epochs=5,
          validation_data=batchCreate(xVal,yVal,50,0),validation_steps=100)

# Step10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.legend('Loss')
plt.xlabel('Epoch')
plt.show()