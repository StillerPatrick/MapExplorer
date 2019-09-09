# MapExplorer

The goal of this project is to recognize text from old maps. 

![mapsample](/docs/sample.jpeg)

Workflow to solve this task

1. generate artificial sample data with random texts
2. train a U-Net that removes the map data, leaving only the text
3. use tesseract OCR to recognize the text of the cleaned map tile  


## Todo

* adapt the exsisting data generator
* write a data loader
* preprocess data (grayscale, opencv denoise) 
* import unet model
* train data cleaning model
* test
* pipe output to [tesseract ocr](https://github.com/tesseract-ocr/)