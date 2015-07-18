import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.naive_bayes as nb
import lxmls.classifiers.multinomial_naive_bayes as mnbb

scr = srs.SentimentCorpus("books")

# mnb = mnbb.MultinomialNaiveBayes()
# params_nb_sc = mnb.train(scr.train_X,scr.train_y)
# y_pred_train = mnb.test(scr.train_X,params_nb_sc)
# acc_train = mnb.evaluate(scr.train_y, y_pred_train)
# y_pred_test = mnb.test(scr.test_X,params_nb_sc)
# acc_test = mnb.evaluate(scr.test_y, y_pred_test)

# print "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(
#     acc_train,acc_test)

import lxmls.readers.simple_data_set as sds
sd = sds.SimpleDataSet(nr_examples=100, g1 = [[-1,-1],1], g2 = [[1,1],1], balance=0.5, split=[0.5,0,0.5])
# sd = scr

import lxmls.classifiers.perceptron as percc
perc = percc.Perceptron()
params_perc_sd = perc.train(sd.train_X,sd.train_y)
y_pred_train = perc.test(sd.train_X,params_perc_sd)
acc_train = perc.evaluate(sd.train_y, y_pred_train)
y_pred_test = perc.test(sd.test_X,params_perc_sd)
acc_test = perc.evaluate(sd.test_y, y_pred_test)
print "Perceptron Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)

perc = percc.Perceptron()
params_perc_sd = perc.train(scr.train_X,sd.train_y)
y_pred_train = perc.test(scr.train_X,params_perc_sd)
acc_train = perc.evaluate(scr.train_y, y_pred_train)
y_pred_test = perc.test(scr.test_X,params_perc_sd)
acc_test = perc.evaluate(scr.test_y, y_pred_test)
print "Perceptron Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)

fig, axis = sd.plot_data()
fig, axis = sd.add_line(fig, axis, params_perc_sd, "Perceptron", "blue")

import lxmls.classifiers.mira as mirac

mira = mirac.Mira()
mira.regularizer = 1.0 # This is lambda
params_mira_sd = mira.train(sd.train_X,sd.train_y)
y_pred_train = mira.test(sd.train_X,params_mira_sd)
acc_train = mira.evaluate(sd.train_y, y_pred_train)
y_pred_test = mira.test(sd.test_X,params_mira_sd)
acc_test = mira.evaluate(sd.test_y, y_pred_test)
print "Mira Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test) 

fig,axis = sd.add_line(fig,axis,params_mira_sd,"Mira","green")
params_mira_sc = mira.train(scr.train_X,scr.train_y)
y_pred_train = mira.test(scr.train_X,params_mira_sc)
acc_train = mira.evaluate(scr.train_y, y_pred_train)
y_pred_test = mira.test(scr.test_X,params_mira_sc)
acc_test = mira.evaluate(scr.test_y, y_pred_test)
print "Mira Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)