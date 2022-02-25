function postSelection(body_part){
   var xhr=new XMLHttpRequest();

   xhr.open("POST",`/select_body_part?selection=${body_part}`,true);
   xhr.send();
}

var button = document.getElementsByClassName("button");

var addSelectClass = function(){
    removeSelectClass();
    this.classList.add('selected');
    postSelection(this.id);
}

var removeSelectClass = function(){
    for (var i =0; i < button.length; i++) {
        button[i].classList.remove('selected')
    }
}

for (var i =0; i < button.length; i++) {
    button[i].addEventListener("click",addSelectClass);
}

var ws = new WebSocket("ws://localhost:8000/websocket_reset_selection");
ws.onmessage = function(event) {
    returnjson = JSON.parse(event.data);
    if (returnjson.message == "reset_ui") {
        removeSelectClass();
    }

}