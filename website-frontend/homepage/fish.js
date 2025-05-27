$(document).ready(function() {
    const profileForm = $('#profileForm');
    const profileList = $('#profileList');
    const fishContainer = $('#fishContainer');
    const sortOrder = $('#sortOrder');
    const fish3D = $('#fish3D');


    
    const savedProfiles = JSON.parse(localStorage.getItem('profiles')) || [];
    let sortedProfiles;

    fish3D.click(function(){
        if(savedProfiles.length<3){
            alert("名前を3つ入力してください");
            return;
        }
        window.location.href = "../unitychatpage/index.html";
    });

    function displayProfiles() {
        profileList.empty();
        const order = sortOrder.val();
        sortedProfiles = [...savedProfiles];
    
        if (order === 'name') sortedProfiles.sort((a, b) => a.name.localeCompare(b.name));
        else if (order === 'kind') sortedProfiles.sort((a, b) => a.kind.localeCompare(b.kind));
    
        sortedProfiles.forEach((profile, index) => {
            const listItem = $('<li></li>').text(`名前: ${profile.name}, 品種: ${profile.kind}`).addClass('list-item');
    
            const deleteButton = $('<button></button>').text('削除').addClass('delete-btn');
            deleteButton.on('click', function(e) {
                e.stopPropagation(); 
                savedProfiles.splice(index, 1); 
                displayProfiles(); 
            });
            listItem.on('click', function() {
                const nextPage = `../profilepage/detail.html?variable=${encodeURIComponent(JSON.stringify(profile))}`;
                window.location.href = nextPage;
            });
            listItem.append(deleteButton);
            profileList.append(listItem);
        });
    }
    
    
    profileForm.on('submit', function(event) {
        event.preventDefault();
        let rem = 0;
        const name = $('#name').val();
        const kind = $('#kind').val();
        const newProfile = {name, kind, birthday:"",activity:"",chara:"",detail:""};
        savedProfiles.forEach((profile)=>{
            if(profile.name==name){
                alert("名前はそれぞれ変えてください。");
                rem = 1;
                return;
            }
        })
        if(rem == 1)return;
        savedProfiles.push(newProfile);
        localStorage.setItem('profiles', JSON.stringify(savedProfiles));

        displayProfiles();

        profileForm[0].reset();
    });

    $(".open-btn").click(function () {
        $(this).toggleClass('btnactive');//.open-btnは、クリックごとにbtnactiveクラスを付与＆除去。1回目のクリック時は付与
        $("#search-wrap").toggleClass('panelactive');//#search-wrapへpanelactiveクラスを付与
        $('#search-text').focus();//テキスト入力のinputにフォーカス
    });
    const searchForm = $('#searchform');
    
    searchForm.on('submit',function(event){
        event.preventDefault();
        profileList.empty();
        const searchText1 = $('#search-text1').val().toLowerCase();
        const searchText2 = $('#search-text2').val().toLowerCase();
        const searchProfiles = sortedProfiles.filter(profile =>{
            if(searchText1==="") return ((profile.name.toLowerCase()==="")|| profile.kind.toLowerCase().includes(searchText2));
            else if(searchText2==="") return (profile.name.toLowerCase().includes(searchText1) || (profile.kind.toLowerCase()===""));
            else return (profile.name.toLowerCase().includes(searchText1) && profile.kind.toLowerCase().includes(searchText2));
        });
        searchProfiles.forEach((profile) => {
            const listItem = $('<li></li>').text(`名前: ${profile.name}, 品種: ${profile.kind}`);
            listItem.on('click', function() {
                const nextPage = `../profilepage/detail.html?variable=${encodeURIComponent(JSON.stringify(profile))}`;
                window.location.href = nextPage;
            });
            profileList.append(listItem);
        });
        //displayProfiles(searchProfiles);
    });



    sortOrder.on('change', displayProfiles); 

    displayProfiles();

    const photoNumber = 5;

    // 魚のイラストをランダムに生成する関数
    function createFish() {
        const fish = $('<div class="fish"></div>');
        const isLeftToRight = Math.random() > 0.5;

        fish.css({
            top: Math.random() * $(window).height() + 'px',
            left: isLeftToRight ? '-50px' : $(window).width() + 'px',
            backgroundImage: `url(./fish-kind/魚のイラスト${Math.ceil(photoNumber * Math.random())}.png)`
        });

        if (isLeftToRight) {
            fish.addClass('flip');
        }

        fishContainer.append(fish);

        animateFish(fish, isLeftToRight);
    }

    // 魚のイラストをアニメーションさせる関数
    function animateFish(fish, isLeftToRight) {
        const direction = isLeftToRight ? '+=' : '-=';
        const distance = $(window).width() + 100;
        const duration = 10000 + Math.random() * 5000;

        fish.animate({
            left: direction + distance + 'px'
        }, duration, 'linear', function() {
            fish.remove();
        });
    }

    // 一定間隔で魚を生成する
    setInterval(createFish, 2000);


    var unit = 100;
    var canvas, context;
    var info = {
        seconds: 0,
        t: 0
    };
    var colors = ['#43c0e4', '#49a2bb', '#4793a8']; // 波の色設定

    function init() {
        canvas = document.getElementById("waveCanvas");
        canvas.width = document.documentElement.clientWidth; // Canvasの幅をウィンドウの幅に合わせる
        canvas.height = 200; // 波の高さ
        context = canvas.getContext("2d");

        update(); // アニメーション開始
    }

    function update() {
        draw();

        info.seconds += 0.01; // 時間の増加量を少し減らしてスムーズに
        info.t = info.seconds * Math.PI;

        requestAnimationFrame(update); // スムーズなアニメーションのために使用
    }

    function draw() {
        context.clearRect(0, 0, canvas.width, canvas.height);

        drawWave(colors[0], 0.5, 3, 0);
        drawWave(colors[1], 0.4, 2, 250);
        drawWave(colors[2], 0.2, 1.6, 100);
    }

    function drawWave(color, alpha, zoom, delay) {
        context.fillStyle = color;
        context.globalAlpha = alpha;
        context.beginPath();
        drawSine(info.t / 0.5, zoom, delay);
        context.lineTo(canvas.width + 10, canvas.height);
        context.lineTo(0, canvas.height);
        context.closePath();
        context.fill();
    }

    function drawSine(t, zoom, delay) {
        var xAxis = Math.floor(canvas.height / 2); // 波の中心軸をCanvasの高さの中央に
        var yAxis = 0;
        var x = t;
        var y = Math.sin(x) / zoom;
        context.moveTo(yAxis, unit * y + xAxis);

        for (var i = yAxis; i <= canvas.width + 10; i += 10) {
            x = t + (-yAxis + i) / unit / zoom;
            y = Math.sin(x - delay) / 3;
            context.lineTo(i, unit * y + xAxis);
        }
    }

    init(); // 初期化・アニメーション開始
});