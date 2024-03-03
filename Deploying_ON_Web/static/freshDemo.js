var stop_flag = 0;
var finalResult_div = getElementById("FinalResult");

function CheckImgExists(imgurl) {
      return new Promise(function (resolve, reject) {
		var ImgObj = new Image();
		ImgObj.src = imgurl;
		ImgObj.onload = function (res) {
			resolve(res);
		    console.log("res is",res);
		}
		ImgObj.onerror = function (err) {
			reject(err);
			console.log("err is",err);
		}
	})
}


function run_CheckExists(){
    while(true){
        CheckImgExists('../saved_imgs/img.jpg')//imgurl here
        .then(() => {onload =
            function f() {
                document.getElementById("FinalResult").innerHTML = '<img src="../saved_imgs/img.jpg" width="400px" height="450px"/>';
            }
        }) //success callback
        .catch(() => {console.log('no');})//fail callback
    }
}


CheckImgExists('../saved_imgs/img.jpg')//imgurl here
.then(() => {onload =
    function f() {
        document.getElementById("FinalResult").innerHTML = '<img src="../saved_imgs/img.jpg" width="400px" height="450px"/>'

    }}) //success callback
.catch(() => {console.log('no');})//fail callback

