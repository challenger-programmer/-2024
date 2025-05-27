$(document).ready(function(){
    var container = document.querySelector("#unity-container");
    var canvas = document.querySelector("#unity-canvas");
    var loadingBar = document.querySelector("#unity-loading-bar");
    var progressBarFull = document.querySelector("#unity-progress-bar-full");
    var fullscreenButton = document.querySelector("#unity-fullscreen-button");
    var warningBanner = document.querySelector("#unity-warning");
    const modal = $('#tutorial-modal');
    const nameInput = $('#fishFormUpdate');
    const savedProfilesCP = JSON.parse(localStorage.getItem('profiles'));
    const FishNamesCP = JSON.parse(localStorage.getItem('FishNames'))||{ fishName1: "あ", fishName2: "あ", fishName3: "あ" };
    let count = 0;
    console.log(FishNamesCP);
    // if (!FishNamesCP || !FishNamesCP.fishName1 || !FishNamesCP.fishName2 || !FishNamesCP.fishName3) {
    //     FishNamesCP = { fishName1: "", fishName2: "", fishName3: "" };
    // }

   
    for(let i=0;i<savedProfilesCP.length;i++){
        if(savedProfilesCP[i].name===FishNamesCP.fishName1)count++;
        else if(savedProfilesCP[i].name===FishNamesCP.fishName2)count++;
        else if(savedProfilesCP[i].name===FishNamesCP.fishName3)count++;
        console.log(1);
    }
    if(FishNamesCP==[]||count<3){
        modal.css('display', 'flex');
        const nameInput1 = $('#name-input1');
        const nameInput2 = $('#name-input2');
        const nameInput3 = $('#name-input3');
        const savedProfiles = JSON.parse(localStorage.getItem('profiles'));


        savedProfiles.forEach((profile) => {
            const option1 = $('<option></option>').val(profile.name).text(profile.name);
            const option2 = $('<option></option>').val(profile.name).text(profile.name);
            const option3 = $('<option></option>').val(profile.name).text(profile.name);
            nameInput1.append(option1);
            nameInput2.append(option2);
            nameInput3.append(option3);
        });
        
    } 
    else modal.hide();
    // ページが読み込まれたらモーダルを表示
    // モーダルを表示
    // 送信ボタンをクリックした時の処理
    
    //unity干渉しないように
    $(document).on('keydown', function(event) {
        // 現在フォーカスされている要素がinputまたはtextareaの場合、Unityへのイベントを無視する
        if ($(document.activeElement).is('input, textarea')) {
            event.stopPropagation();
        }
    });
    

    function unityShowBanner(msg, type) {
        function updateBannerVisibility() {
            warningBanner.style.display = warningBanner.children.length ? 'block' : 'none';
        }
        var div = document.createElement('div');
        div.innerHTML = msg;
        warningBanner.appendChild(div);
        if (type == 'error') div.style = 'background: red; padding: 10px;';
        else {
            if (type == 'warning') div.style = 'background: yellow; padding: 10px;';
            setTimeout(function () {
                warningBanner.removeChild(div);
                updateBannerVisibility();
            }, 5000);
        }
        updateBannerVisibility();
    }

    var buildUrl = "Build";
    var loaderUrl = buildUrl + "/unity-build8.loader.js";
    var config = {
        dataUrl: buildUrl + "/unity-build8.data",
        frameworkUrl: buildUrl + "/unity-build8.framework.js",
        codeUrl: buildUrl + "/unity-build8.wasm",
        streamingAssetsUrl: "StreamingAssets",
        companyName: "DefaultCompany",
        productName: "VR_fast",
        productVersion: "0.1",
        showBanner: unityShowBanner,
    };

    if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
        var meta = document.createElement('meta');
        meta.name = 'viewport';
        meta.content = 'width=device-width, height=device-height, initial-scale=1.0, user-scalable=no, shrink-to-fit=yes';
        document.getElementsByTagName('head')[0].appendChild(meta);
        container.className = "unity-mobile";
        canvas.className = "unity-mobile";
    } else {
        canvas.style.width = "960px";
        canvas.style.height = "600px";
    }

    loadingBar.style.display = "block";
    var script = document.createElement("script");
    script.src = loaderUrl;

    script.onload = () => {
        createUnityInstance(canvas, config, (progress) => {
            progressBarFull.style.width = 100 * progress + "%";
        }).then((unityInstance) => {
            loadingBar.style.display = "none";
            fullscreenButton.onclick = () => {
                unityInstance.SetFullscreen(1);
            };

            // unityInstanceの取得をここで行う
            setupForms(unityInstance);
        }).catch((message) => {
            alert(message);
        });
    };

    document.body.appendChild(script);//{roomName1: '魚1', roomName2: '魚2', roomName3: '魚3'}
    

    $('#back').on('click', function(){
        window.location.href = `../homepage/index.html`;
    })
   
    



    // フォームの処理をまとめる
    function setupForms(unityInstance) {
        nameInput.on('submit', function(event) {
            event.preventDefault();
            const nameInput1 = $('#name-input1').val();
            const nameInput2 = $('#name-input2').val();
            const nameInput3 = $('#name-input3').val();
            if (nameInput1==nameInput2||nameInput1==nameInput3||nameInput3==nameInput2) {
                alert("名前が同じものがあります"); // エラーメッセージを表示
            }
            else {
                // モーダルを非表示
                const fishNamesWrapper = { fishNames: [nameInput1, nameInput2, nameInput3] };
                const fishNamesJson = JSON.stringify(fishNamesWrapper);
                localStorage.setItem('FishNames', JSON.stringify({fishName1:nameInput1,fishName2:nameInput2,fishName3:nameInput3}));
                unityInstance.SendMessage('Dropdown2', 'ReceiveFishDataFromJS', fishNamesJson); modal.hide();
            }
        });
    






        const roomForm = $('#roomForm');
        const fishForm = $('#fishForm');
        const chatForm = $('#chatForm');
        const fishName1 = $('#fishName1');
        const fishName2 = $('#fishName2');
        const fishName3 = $('#fishName3');
        const RoomNames = JSON.parse(localStorage.getItem('RoomNames')) || {roomName1: '部屋1', roomName2: '部屋2', roomName3: '部屋3'};
        const FishNames = JSON.parse(localStorage.getItem('FishNames')) || {fishName1: '魚1', fishName2: '魚2', fishName3: '魚3'};
        console.log(RoomNames);
        
        const displayRoomNames = { roomNames: [RoomNames.roomName1, RoomNames.roomName2, RoomNames.roomName3] };
        const displayFishNames = { fishNames: [FishNames.fishName1, FishNames.fishName2, FishNames.fishName3] };
        const displayRoomNamesJson = JSON.stringify(displayRoomNames);
        const displayFishNamesJson = JSON.stringify(displayFishNames);
        unityInstance.SendMessage('Dropdown', 'ReceiveRoomDataFromJS', displayRoomNamesJson);
        unityInstance.SendMessage('Dropdown2', 'ReceiveFishDataFromJS', displayFishNamesJson);

        const savedProfiles = JSON.parse(localStorage.getItem('profiles')) || [];
        let savedProfileName = [];

        savedProfiles.forEach((profile) => {
            savedProfileName.push(profile.name);
        });

        console.log(savedProfileName);

        savedProfileName.forEach((name) => {
            const option1 = $('<option></option>').val(name).text(name);
            const option2 = $('<option></option>').val(name).text(name);
            const option3 = $('<option></option>').val(name).text(name);
            fishName1.append(option1);
            fishName2.append(option2);
            fishName3.append(option3);
        });
        
        //JS->unity
        roomForm.on('submit', function (event) {
            event.preventDefault();
            const roomName1 = $('#roomName1').val();
            const roomName2 = $('#roomName2').val();
            const roomName3 = $('#roomName3').val();
            const roomNamesWrapper = { roomNames: [roomName1, roomName2, roomName3] };
            const roomNamesJson = JSON.stringify(roomNamesWrapper);

            if(roomName1==roomName2||roomName1==roomName3||roomName2==roomName3){
                alert("部屋の名前に同じものがあります。")
                return;
            }

            localStorage.setItem('RoomNames', JSON.stringify({roomName1,roomName2,roomName3}));

            unityInstance.SendMessage('Dropdown', 'ReceiveRoomDataFromJS', roomNamesJson);
        });

        fishForm.on('submit', function (event) {
            event.preventDefault();
            const fishName1 = $('#fishName1').val();
            const fishName2 = $('#fishName2').val();
            const fishName3 = $('#fishName3').val();
            if(fishName1==fishName2||fishName1==fishName3||fishName2==fishName3){
                alert("お魚の名前に同じものがあります。お魚が悲しみます。")
                return;
            }
            const fishNamesWrapper = { fishNames: [fishName1, fishName2, fishName3] };
            const fishNamesJson = JSON.stringify(fishNamesWrapper);
            localStorage.setItem('FishNames', JSON.stringify({fishName1,fishName2,fishName3}));
            unityInstance.SendMessage('Dropdown2', 'ReceiveFishDataFromJS', fishNamesJson);
        });


        //jsで魚としゃべる。
        chatForm.on('submit',function(event){
            event.preventDefault();
            const chatContent = $('#chatFish').val();
            unityInstance.SendMessage('Text (TMP)1','ReceiveChatDataFromJS',chatContent);
        });
        //js 魚としゃべるのをやめる
        $('#finishChat').on('click',function(event){
            event.preventDefault();
            unityInstance.SendMessage('Text (TMP)1','FinishFromJS');
        });


        $('#changeAcitv').on('submit',function(){
            const changeActiv1 = $('#changeActiv1').val();
            const changeActiv2 = $('#changeActiv2').val();
            const changeActiv3 = $('#changeActiv3').val();
            const fishNames = JSON.parse(localStorage.getItem('FishNames'));
            let genki1,genki2,genki3;
            // if(data.activ_syubun==1)activity = "元気がない";
            //         else if(data.activ_syubun==2)activity = "あまり元気がない";
            //         else if(data.activ_syubun==3)activity = "少し元気";
            //         else activity = "元気いっぱい";
            if(changeActiv1=="元気いっぱい"||changeActiv1=="少し元気")genki1=2;
            else if(changeActiv1=="あまり元気がない")genki1=1;
            else if(changeActiv1=="元気がない")genki1=0;
            
            if(changeActiv2=="元気いっぱい"||changeActiv2=="少し元気")genki2=2;
            else if(changeActiv2=="あまり元気がない")genki2=1;
            else if(changeActiv2=="元気がない")genki2=0;
            
            if(changeActiv3=="元気いっぱい"||changeActiv3=="少し元気")genki3=2;
            else if(changeActiv3=="あまり元気がない")genki3=1;
            else if(changeActiv3=="元気がない")genki3=0;

            savedProfiles.forEach((profile)=>{
                if(profile.name == fishNames.fishName1){
                    unityInstance.SendMessage('Text (TMP)1','Chara1FromJS',"あなたは"+profile.chara+"で"+ profile.name+"という名前です。今日あなたは"+changeActiv1+"日本語で友達のように話してください。また短い文章で答えてください。");
                    unityInstance.SendMessage('FishV1-2','ActivFromJS',genki1);
                }
                else if(profile.name == fishNames.fishName2){
                    unityInstance.SendMessage('Text (TMP)1','Chara1FromJS',"あなたは"+profile.chara+"で"+ profile.name+"という名前です。今日あなたは"+changeActiv2+"日本語で友達のように話してください。また短い文章で答えてください。");
                    unityInstance.SendMessage('FishV2-2','ActivFromJS',genki2);
                }
                else if(profile.name == fishNames.fishName3){
                    unityInstance.SendMessage('Text (TMP)1','Chara1FromJS',"あなたは"+profile.chara+"で"+ profile.name+"という名前です。今日あなたは"+changeActiv3+"日本語で友達のように話してください。また短い文章で答えてください。");
                    unityInstance.SendMessage('FishV3-2','ActivFromJS',genki3);
                }
            })

        });
        $('#changeChara').on('submit',function(){
            const changeChara1 = $('changeChara1').val();
            const changeChara2 = $('changeChara2').val();
            const changeChara3 = $('changeChara3').val();
            const fishNames = JSON.parse(localStorage.getItem('FishNames'));
            savedProfiles.forEach((profile)=>{
                if(profile.name==fishNames.fishName1)unityInstance.SendMessage('Text (TMP)1','Chara1FromJS',"あなたは"+changeChara1+"で、この性格に基づいて口調を変えてください。あなたは"+ profile.name+"という名前です。今日あなたは"+profile.activity+"です。日本語で友達のように話してください。また短い文章で答えてください。");
                if(profile.name==fishNames.fishName2)unityInstance.SendMessage('Text (TMP)1','Chara1FromJS',"あなたは"+changeChara2+"で"+ profile.name+"という名前です。今日あなたは"+profile.activity+"です。日本語で友達のように話してください。また短い文章で答えてください。");
                if(profile.name==fishNames.fishName3)unityInstance.SendMessage('Text (TMP)1','Chara1FromJS',"あなたは"+changeChara3+"で"+ profile.name+"という名前です。今日あなたは"+profile.activity+"です。日本語で友達のように話してください。また短い文章で答えてください。");
            })
        })

        //unityにデータをアップデート
        {
            const fishNames = JSON.parse(localStorage.getItem('FishNames'))||{fishName1: '魚1', fishName2: '魚2', fishName3: '魚3'};
            const aiData = JSON.parse(localStorage.getItem('AIData'));
            
            let genki;
            savedProfiles.forEach((profile)=>{
                if(profile.name==fishNames.fishName1){
                    unityInstance.SendMessage('Text (TMP)1','Chara1FromJS',"あなたは"+profile.chara+"で"+ profile.name+"という名前です。今日あなたは"+profile.activity+"です。日本語で友達のように話してください。また短い文章で答えてください。");
                    if(aiData.activ_syubun==1||aiData.activ_syubun==0)genki=2;
                    else if(aiData.activ_syubun==2)genki=1;
                    else genki=0;
                    unityInstance.SendMessage('FishV1-2','ActivFromJS',genki);
                }
                else if(profile.name==fishNames.fishName2){
                    unityInstance.SendMessage('Text (TMP)1','Chara2FromJS',"あなたは"+profile.chara+"で"+ profile.name+"という名前です。今日あなたは"+profile.activity+"です。日本語で友達のように話してください。また短い文章で答えてください。");
                    if(aiData.activ_ryukin==1||aiData.activ_ryukin==0)genki=2;
                    else if(aiData.activ_ryukin==2)genki=1;
                    else genki=0;
                    unityInstance.SendMessage('FishV2-2','ActivFromJS',genki);
                }
                else if(profile.name==fishNames.fishName3){
                    unityInstance.SendMessage('Text (TMP)1','Chara3FromJS',"あなたは"+profile.chara+"で"+ profile.name+"という名前です。今日あなたは"+profile.activity+"です。日本語で友達のように話してください。また短い文章で答えてください。");
                    if(aiData.activ_demekin==1||aiData.activ_demekin==0)genki=2;
                    else if(aiData.activ_demekin==2)genki=1;
                    else genki=0;
                    unityInstance.SendMessage('FishV3-2','ActivFromJS',genki);
                }
            })
        }


        $('#unity-canvas').on('mouseenter', function() {
            // Unityにカメラを有効にするメッセージを送信
            unityInstance.SendMessage('Room1Camera','OnMessageReceived','ENABLE_CAMERA');
            unityInstance.SendMessage('Room2Camera','OnMessageReceived','ENABLE_CAMERA');
            unityInstance.SendMessage('Room3Camera','OnMessageReceived','ENABLE_CAMERA');
          });
      
          // マウスがUnity部分から出たとき
          $('#unity-canvas').on('mouseleave', function() {
            // Unityにカメラを無効にするメッセージを送信
            unityInstance.SendMessage('Room1Camera','OnMessageReceived','DISABLE_CAMERA');
            unityInstance.SendMessage('Room2Camera','OnMessageReceived','DISABLE_CAMERA');
            unityInstance.SendMessage('Room3Camera','OnMessageReceived','DISABLE_CAMERA');
          });

    };


    //AIで分析
    const infoUpdate = $('#infoUpdate');
    infoUpdate.on('click',function(){

        const data = JSON.parse(localStorage.getItem('AIData')) || {
            average_distance_syu: 5000,
            average_distance_ryu: 5000,
            average_distance_deme: 5000,
            range_distance_syu: 2000,
            range_distance_ryu: 2000,
            range_distance_deme: 2000,
            hide_syubun_level: 1,
            hide_ryukin_level: 1,
            hide_demekin_level: 1,
            fish_gather: [[0,0,0],[0,0,0],[0,0,0]]
        };
        
        
        fetch('https://192.168.164.135:8000/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)  // data を JSON 形式に変換して送信
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();  // レスポンスを JSON 形式で解析
        })
        .then(data => {
            console.log("Processed data received from Python:", data);
            
            // 受け取ったデータをローカルストレージに保存
            localStorage.setItem('AIData', JSON.stringify(data));
            
            const savedProfiles = JSON.parse(localStorage.getItem('profiles')) || [];
            const fishNames = JSON.parse(localStorage.getItem('FishNames')) || [];
            
            let changeProfiles = [];
            savedProfiles.forEach((profile) => {
                let activity;
                let chara1;
                let chara2;
                let chara;
                if(profile.name==fishNames.fishName1){
                    if(data.activ_syubun==1)activity = "元気がない";
                    else if(data.activ_syubun==2)activity = "あまり元気がない";
                    else if(data.activ_syubun==3)activity = "少し元気";
                    else activity = "元気いっぱい";
                    const gather_syubun = (data.fish_gather[0][1] + data.fish_gather[0][2])/2
                    if(gather_syubun>0.65)chara1 = "社交的で";
                    else if(gather_syubun>0.35)chara1 = "柔軟に適応できる"
                    else chara1 = "慎重な";

                    if(data.hide_syubun_level==3){
                        chara2 = "一人が好きな";
                        if(chara1 == "社交的で")chara1 = "社交的だけど";
                    }
                    else if(data.hide_syubun_level==2)chara2 = "謙虚な";

                    else {
                        chara2 = "自身に満ち溢れた";
                        if(chara1 == "慎重な")chara1 = "慎重だけど";
                    }
                    chara = chara1 + chara2 + "子";
                }
                else if(profile.name == fishNames.fishName2){
                    if(data.activ_ryukin==1)activity = "元気がない";
                    else if(data.activ_ryukin==2)activity = "あまり元気がない";
                    else if(data.activ_ryukin==3)activity = "少し元気";
                    else activity = "元気いっぱい";
                    const gather_ryukin = (data.fish_gather[0][1] + data.fish_gather[0][2])/2
                    if(gather_ryukin>0.65)chara1 = "社交的で";
                    else if(gather_ryukin>0.35)chara1 = "柔軟に適応できる"
                    else chara1 = "慎重な";

                    if(data.hide_ryukin_level==3){
                        chara2 = "一人が好きな";
                        if(chara1 == "社交的で")chara1 = "社交的だけど";
                    }
                    else if(data.hide_ryukin_level==2)chara2 = "謙虚な";

                    else {
                        chara2 = "自身に満ち溢れた";
                        if(chara1 == "慎重な")chara1 = "慎重だけど";
                    }
                    chara = chara1 + chara2 + "子";
                }
                else if(profile.name == fishNames.fishName3){
                    if(data.activ_demekin==1)activity = "元気がない";
                    else if(data.activ_demekin==2)activity = "あまり元気がない";
                    else if(data.activ_demekin==3)activity = "少し元気";
                    else activity = "元気いっぱい";
                    const gather_demekin = (data.fish_gather[0][1] + data.fish_gather[0][2])/2
                    if(gather_demekin>0.65)chara1 = "社交的で";
                    else if(gather_demekin>0.35)chara1 = "柔軟に適応できる"
                    else chara1 = "慎重な";

                    if(data.hide_demekin_level==3){
                        chara2 = "一人が好きな";
                        if(chara1 == "社交的で")chara1 = "社交的だけど";
                    }
                    else if(data.hide_demekin_level==2)chara2 = "謙虚な";

                    else {
                        chara2 = "自身に満ち溢れた";
                        if(chara1 == "慎重な")chara1 = "慎重だけど";
                    }
                    chara = chara1 + chara2 + "子";
                }
                //{name, kind, birthday:"",activity:"",chara:"",detail:""}
                changeProfiles.push({
                    name: profile.name,
                    kind: profile.kind,
                    birthday: profile.birthday,
                    activity: activity,
                    chara: chara,
                    detail: profile.detail
                    
                });
                localStorage.setItem('profiles', JSON.stringify(changeProfiles));
            });
            
        });     
    });
});
