# ipi-deep-event-detection

Event identification in Polish, based on deep neural networks.

## Running:

``src/event_detector.py``

## Help:

``src/event_detector.py --help``

## Training the model:
 
``python src/event_detector.py -t -c config/config.json -m models/model_multi_class -i batch:ccl ../sytuacje/index_events.txt``

## Evaluation using cross validation

``python src/event_detector.py -e -c config/config2.json -m models/model_multi_class_cv -i batch:ccl ../sytuacje/index_events_folds_10.list ``

## Event detection (pipe mode):

``python src/event_detector.py -c config/config.json -m models/model_multi_class -i batch:ccl ../sytuacje/index_events_test.txt out/out.txt``

