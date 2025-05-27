$(document).ready(function() {
    const urlParams = new URLSearchParams(window.location.search);
    const receivedVariable = JSON.parse(decodeURIComponent(urlParams.get('variable')));
    
    if(receivedVariable.name != "記入されていません") $('#name').val(receivedVariable.name);
    if(receivedVariable.kind != "記入されていません") $('#kind').val(receivedVariable.kind);
    if(receivedVariable.birthday != "記入されていません") $('#birthday').val(receivedVariable.birthday);
    if(receivedVariable.activity != "記入されていません") $('#activity').val(receivedVariable.activity);
    if(receivedVariable.chara != "記入されていません") $('#chara').val(receivedVariable.chara);
    if(receivedVariable.detail != "記入されていません") $('#detail').val(receivedVariable.detail);

    let profiles = JSON.parse(localStorage.getItem('profiles')) || [];
    
    const button = $('#button');
    button.on('click', function(){
        const name = $('#name').val();
        const kind = $('#kind').val(); 
        const birthday = $('#birthday').val();
        // 活発度と性格は受信した値をそのまま使用
        const activity = receivedVariable.activity;
        const chara = receivedVariable.chara;
        const detail = $('#detail').val();
        
        const newVariable = {name, kind, birthday, activity, chara, detail};

        profiles = profiles.map(profile => {
            if (profile.name == receivedVariable.name && profile.kind == receivedVariable.kind) {
                return {name, kind, birthday, activity, chara, detail};
            }
            return profile;
        });

        localStorage.setItem('profiles', JSON.stringify(profiles));

        const nextPage = `../profilepage/detail.html?variable=${encodeURIComponent(JSON.stringify(newVariable))}`;
        window.location.href = nextPage;
    });
});
