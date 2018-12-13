

class RNet(models.Sequential):
    """
        Using pre-trained ResNet50 without head
    """
    def __init__(self):
        super(RNet, self).__init__()

        sizes = [(200,200,3), (7,7,2048), 1024, 37]

        # Load the pre-trained base model
        base = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=sizes[0])
        # Freeze the layers except the last ones
        # for layer in base.layers[:-4]:
        #     layer.trainable = False

        self.add(base)
        self.add(layers.Flatten())
        self.add(layers.Dense(sizes[-2], activation='relu'))
        self.add(layers.Dense(sizes[-1], activation='sigmoid'))
rnet = RNet()
rnet.summary()

general_class_train_gen = generator.flow_from_directory(subset='training',   directory='../dataset_v2/train/divided/general_class', **flow_args)
general_class_valid_gen = generator.flow_from_directory(subset='validation', directory='../dataset_v2/train/divided/general_class', **flow_args)

large_vehicle_train_gen = generator.flow_from_directory(subset='training',   directory='../dataset_v2/train/divided/large_vehicle', **flow_args)
large_vehicle_valid_gen = generator.flow_from_directory(subset='validation', directory='../dataset_v2/train/divided/large_vehicle', **flow_args)

small_vehicle_gen = generator.flow_from_directory(directory='../dataset_v2/train/divided/small_vehicle', **flow_args)
color_gen         = generator.flow_from_directory(directory='../dataset_v2/train/divided/color', **flow_args)


train_gen = large_vehicle_train_gen
valid_gen = large_vehicle_valid_gen

x_batch, y_batch = next(large_vehicle_train_gen)
for i in range (5):
    image = x_batch[i]
    image = image.astype(np.float)
    plt.imshow(image)
    plt.show()

const = -1/(Nm.sum() + len(w))

_EPSILON = K.epsilon()

def weighted_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(w * y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

# shape = (6,7)
# y_a = np.random.random(shape)
# y_b = np.random.random(shape)
# K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))

def WeightedCELoss(ys, ts):
    return const * (w * ys * tf.log(ts) + (1 - ys) * tf.log(1 - ts))


# get_tag_ids = lambda x: int(re.split(r'[\_.]', x)[-2])
# tag_ids = list(map(get_tag_ids, valid_gen.filenames))
# train = train.reset_index('image_id',drop=True)
# targets = train.ix[tag_ids]
model = MNet()
model.load_weights('../mnet100.h5')

# model = models.load_model('../mnet100.h5', custom_objects={'MNet':MNet})

# Get predictions
predictions_proba = model.predict_generator(valid_gen, steps=len(valid_gen))
# predictions = predictions_proba > 0.1

predictions = list(map(np.argmax, predictions_proba))
targets = valid_gen.classes

AP = np.zeros(len(valid_gen.class_indices))
for category in np.unique(valid_gen.classes):
    ts = valid_gen.classes==category
    ys = np.array(predictions) == category
    AP[category] = APScore(ys, ts)
AP
np.mean(AP)
    
np.save('predictions_proba.npy', predictions_proba)
np.save('predictions.npy', predictions)
predictions = np.load('predictions.npy')

test_predictions_proba = model.predict_generator(test_gen, steps=len(test_gen))
# np.save('test_predictions_proba.npy', test_predictions_proba)

# Get targets
train = pd.read_csv('../dataset_v2/train.csv')
train = train.replace(' ', '_', regex=True)
train = train.replace('/', '_', regex=True)
train = train.drop(['p1_x', 'p_1y', ' p2_x', ' p2_y', ' p3_x', ' p3_y', ' p4_x', ' p4_y'], axis=1)
for category in ['general_class', 'sub_class', 'color']:
    train[category] = train[category].astype('category')
train = pd.get_dummies(train, prefix='', prefix_sep='')
train = train.groupby(['image_id', 'tag_id']).first()
train = train.reindex(sorted(train.columns), axis=1)
targets = (train.values == 1) # Remove -1s

def APScore(ys, ts):
    TP = 0; FP = 0; total = 0
    for (y,t) in zip(ys,ts):
        TP += y and t
        FP += y and not t
        total += (TP / (TP + FP)) if y and t else 0
    # for k in range(1,1+len(ts)):
    #     total += pscore(ts[:k], ys[:k]) * ts[k-1]
    return total / sum(ts)

assert(predictions.shape == targets.shape)
N, M = targets.shape
AP = np.zeros(M)
for category in range(M):
    AP[category] = APScore(predictions[:,category], targets[:,category])
MAP = np.mean(AP)

MAP
AP



# train_gen = generator.flow_from_directory(directory='../dataset_v2/train/classes', **flow_args)
train_gen = generator.flow_from_directory(directory='../dataset_v2/train/divided/large_vehicle', **flow_args)
# valid_gen = generator.flow_from_directory(directory='../dataset_v2/train/cropped', **flow_args)
test_gen = generator.flow_from_directory(directory='../dataset_v2/test/', **flow_args, shuffle=False)

def get_loss_weights(generator):
    Nm = np.array(sorted(Counter(generator.classes).items()))[:,1]
    w0 = np.maximum(Nm.astype(np.float)/generator.samples, 0.1)
    return dict(enumerate( (1 - w0) / w0 ))

Y_train = train_gen.classes
weights = class_weight.compute_class_weight('balanced'
                                               ,np.unique(Y_train)
                                               ,Y_train)



target_dir = '../models/'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

checkpoint = lambda category, period: callbacks.ModelCheckpoint(
    filepath=f'{target_dir}{category}.hdf5', 
    monitor='val_acc', 
    period=period, 
    save_weights_only=True,
    save_best_only=True,
)