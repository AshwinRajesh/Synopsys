
# # Predicting endangered status of animals


import graphlab

endangeredAnimalsTrainData = graphlab.SFrame.read_csv('EndangeredTrainingData.csv')

endangeredAnimalsTrainData.head()

endangeredAnimalsTestData = graphlab.SFrame.read_csv('EndangeredTestData.csv')

endangeredAnimalsTestData.head()

animal_features_nobiome=['Habitat', 'Position on Food Chain','Cause of Endangerment','Reproductivity','Lifespan (years)', 'Movement Behavior']

animal_extinction_model_nobiome=graphlab.logistic_classifier.create(endangeredAnimalsTrainData
                                                    , target='Extinct'
                                                    , features=animal_features_nobiome
                                                    , validation_set=endangeredAnimalsTestData)

animal_extinction_model_nobiome.evaluate(endangeredAnimalsTestData)

animal_extinction_model_nobiome['coefficients']

graphlab.canvas.set_target('ipynb')

animal_extinction_model_nobiome.show(view='Evaluation')

endangeredAnimalsPredictArray = graphlab.SFrame.read_csv('EndangeredPredictData.csv')

endangeredAnimalsPredictArray.head()

endangeredAnimalsPredictArray['extinct_probability'] = animal_extinction_model_nobiome.predict(endangeredAnimalsPredictArray
                                                , output_type='probability')

endangeredAnimalsPredictArray = endangeredAnimalsPredictArray.sort('extinct_probability', ascending=False)

endangeredAnimalsPredictArray[0]

endangeredAnimalsPredictArray[-1]

endangeredAnimalsPredictArray.export_csv('endangered_animal_probability.csv')