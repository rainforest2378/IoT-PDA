import keras.backend as K
from keras import layers,losses
from keras.engine.topology import Layer
import model

source_cnn,source_feature=model.build_source_cnn()
target_cnn,target_feature=model.build_target_cnn()
discriminator1=model.build_domain_classifier()
discriminator2=model.build_domain_classifier_0()

length=75

t_origin = layers.Input(name='input4', shape=(length ,1))

cgan_input = t_origin
tf=target_feature(cgan_input)
generator_output=tf
output = discriminator2(generator_output)
target_fe_trainer = Model(cgan_input,output)
target_fe_trainer.compile(optimizer='adam',loss=['binary_crossentropy'])
discriminator1.compile(optimizer='adam',loss=['binary_crossentropy'])
discriminator2.compile(optimizer='adam',loss=['binary_crossentropy'])



def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        #np.random.shuffle(idxes)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x
source_train_data_generator=get_data_generator(source_x_train)
target_train_data_generator=get_data_generator(target_x_train)

niter=50

for i in range(niter):
    ################
    
    source_x_train_batch = source_train_data_generator.__next__()
    target_x_train_batch = target_train_data_generator.__next__()
    
    source_cnn.trainable=False
    target_cnn.trainable=False
    
    discriminator1.trainable=True
    discriminator2.trainable=True
    
    sf=source_feature.predict(source_x_train_batch)
    tf=target_feature.predict(target_x_train_batch)
    
    sf_tf=np.concatenate([sf,tf],axis=0)

    label1=np.concatenate([np.ones(len(sf)),np.zeros(len(tf))],axis=0)
    loss1=discriminator1.train_on_batch(sf_tf,label1)
    
    ws=1.0/(discriminator1.predict(sf)+1.0)

    
    ws_sf_tf=np.concatenate([sf*ws,tf],axis=0)
    label2=np.concatenate([np.ones(len(sf)),np.zeros(len(tf))],axis=0)
    loss2=discriminator2.train_on_batch(ws_sf_tf,label2)
    
    
    #########################
    discriminator1.trainable=False
    discriminator2.trainable=False
    target_feature.trainable=True
    source_cnn.trainable=False
    
    
    loss_fe=target_fe_trainer.train_on_batch(target_x_train_batch,np.ones(len(tf)))
    if i % 25 == 0:
        print(f'niter: {i+1}, fe_loss: {loss_fe}, d_loss:{loss1},{loss2}')
