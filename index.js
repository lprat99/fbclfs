var express = require('express');
var app = express();
var fs = require("fs");
var bodyParser = require('body-parser');
var request = require('request');
var resolve = require('path').resolve

const fileUpload = require('express-fileupload');
var jimp = require('jimp');
var oxr = require('onnxruntime-web');
var session=0;
app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({
         limit: '50mb',
         extended: true
      }));
      
function mean(arr){
var total = 0;
for(var i = 0; i < arr.length; i++) {
    total += arr[i];
}

return total / arr.length;
}
function stdev(array){
  const n = array.length
  const mean = array.reduce((a, b) => a + b) / n
  return Math.sqrt(array.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n)
}
async function runModel(model, preprocessedData){
    const start = new Date();
    try {
      const feeds = {};
      feeds[model.inputNames[0]] = preprocessedData;
      const outputData = await model.run(feeds);
      const end = new Date();
      const inferenceTime = (end.getTime() - start.getTime());
      const output = outputData[model.outputNames[0]];
      return [output, inferenceTime];
    } catch (e) {
      console.error("error");
     // throw new Error();
    }
  }

function imageDataToTensor(data, dims){
  console.log("im to tensor called");
    const [R, G, B] = new Array([], [], []);
    for (let i = 0; i < data.length; i += 4) {
      R.push(data[i]);
      G.push(data[i + 1]);
      B.push(data[i+2]);
    }
    meanR=mean(R);meanG=mean(G);meanB=mean(B);stdevR=stdev(R);stdevG=stdev(G);stevB=stdev(B)
    for (let i = 0; i<R.length;i+=1){
      R[i]=(R[i]/255.0-0.485)/0.229; G[i]=(G[i]/255.0-0.456)/0.224;B[i]=(B[i]/255.0-0.406)/0.225;
      if(i<100){console.log(R[i]+" i "+i);}
    }
    const transposedData = R.concat(G).concat(B);
    let i, l = transposedData.length; // length, we need this for the loop
    const float32Data = new Float32Array( 224 * 224 *3); // create the Float32Array for output
    for (i = 0; i < l; i++) {float32Data[i] = transposedData[i] ;} // convert to float
      const inputTensor = new oxr.Tensor("float32", float32Data, dims);
    
    return inputTensor;
  }
  
  async function load_image(buf,width = 224, height= 224){
    var imageData = await jimp.read(buf).then((imageBuffer) => {
    return imageBuffer.resize(width, height);
    });
    var data = imageDataToTensor(imageData.bitmap.data, [1,3,224,224])
    return data;
    }
  
    async function inference(buf){
    //const [output, time] = await runModel(session, img);
      try {
      console.log("iference started");
        if(session==0){
          console.log("resolve : "+resolve('./model/facebeautyV2.onnx'))
       session = await oxr.InferenceSession
                          .create('https://dl.dropboxusercontent.com/s/74oyfue545pfhht/facebeautyV2.onnx');
        }
      }catch (e) {
      console.log("error : "+e);
     // throw new Error();
    }
        var img=await load_image(buf);
        console.log("model loaded ?" +session)
        const [res, time] =  await runModel(session, img);
        console.log("res.data "+res);
        return res.data;
    
}



app.get('/', function(req, res){
fs.readFile('./index2.html', function(err, data) {
    //var session =  oxr.InferenceSession
    ///                      .create('https://filebin.net/ojy16zweqw6thv1j/facebeautyV2.onnx');
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write(data);
    return res.end();
  });
    
});

app.get('/model', function(req, res){
  var buf = Buffer.from(JSON.stringify(req.body.content),"base64");
  results = inference(buf);
  console.log('results')
fs.readFile('./index2.html', function(err, data) {
    //var session =  oxr.InferenceSession
    ///                      .create('https://filebin.net/ojy16zweqw6thv1j/facebeautyV2.onnx');
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write(data);
    return res.end();
  });
    
});


app.post('/getfaces', (req, res) => {
  
  request.post(
    'https://vision.googleapis.com/v1/images:annotate?key=AIzaSyAOTK1sI1DKmQzrwnvHnlsuS8iCfAb1ryg',
    { json: req.body},
    function (error, response, body) {
        if (!error && response.statusCode == 200) {
            console.log(body);
            console.log("resp : "+response)
            res.send(response);

        }else{
          console.log("error : "+error)
        }
    }
);

    
});
app.post('/test', (req, res) => {
  
  var buf = Buffer.from(JSON.stringify(req.body.content),"base64");
  results = inference(buf);
  console.log("received");
  results.then(function(value){
    console.log("promise done : "+value);
      res.send(value);

  });

})



app.listen(8080);
