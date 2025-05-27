using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Unity.VisualScripting;
using System.Runtime.InteropServices;

public class BehaviourScript : MonoBehaviour
{ 
    public AnimationClip[] clips;
    public AnimationClip[] fins;
    public AnimationClip[] positions;
    public float interval = 2.0f;
    public int genki = 1;
    private Animator animator;
    private float timer = 0.0f;
    private int currentClipIndex;
    private AnimatorStateInfo stateInfo;
    public float scalePosition = 1.0f;
    public TextMeshPro talk;
    private int remfin = 0; // Updateから移動してクラスのフィールドとして宣言
    private int rempos = 0; // Updateから移動してクラスのフィールドとして宣言

    void Start()
    {
        animator = GetComponent<Animator>();
        animator.Play(fins[genki].name, 0, 0f); // レイヤー0でFin-movementを再生
        stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        animator.Play(positions[genki].name, 2, 0f); // レイヤー2でRandomPositionを再生
        currentClipIndex = 2 * genki;
        animator.Play(clips[currentClipIndex].name, 1, 0f);
        //SendFishData(this.gameObject.name,genki+1);
    }

    
    //public void SendFishData(string name, int genki)
    //{
    //    var activityData = new
    //    {
    //        objectId = name,
    //        activity = genki
    //    };
    //    string jsonData = JsonUtility.ToJson(activityData);
    //    SendMessageToJS("SendFishDataToJS", jsonData);
    //}

    //[DllImport("__Internal")]
    //private static extern void SendMessageToJS(string functionName, string data);

    private void PlayNextAnimation()
    {
        animator.Play(clips[currentClipIndex].name, 1, 0f); // レイヤー1で再生
        currentClipIndex = ((currentClipIndex + 1) % 2) + 2 * genki;
    }

    void Update()
    {
        timer += Time.deltaTime;

        if (remfin == 0 && talk.text != "")
        {
            animator.Play("StopFin", 0, 0f);
            remfin = 1;
        }
        else if (remfin == 1 && talk.text == "")
        {
            animator.Play(fins[genki].name, 0, 0f);
            remfin = 0;
        }

        if (rempos == 0 && talk.text != "")
        {
            animator.Play("StopRandom", 2, 0f);
            rempos = 1;
        }
        else if (rempos == 1 && talk.text == "")
        {
            animator.Play(positions[genki].name, 2, 0f);
            rempos = 0;
        }

        // 現在のクリップの長さを考慮し、次のクリップに移るタイミングを設定
        if ((timer >= (clips[currentClipIndex].length / stateInfo.speed) + interval) && talk.text == "")
        {
            PlayNextAnimation();
            timer = 0.0f;
        }

        stateInfo = animator.GetCurrentAnimatorStateInfo(1); // レイヤー1のステート情報を更新
    }
    public void ActivFromJS(int activ)
    {
        genki = activ;
        animator.Play(fins[genki].name, 0, 0f); // レイヤー0でFin-movementを再生
        stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        animator.Play(positions[genki].name, 2, 0f); // レイヤー2でRandomPositionを再生
        currentClipIndex = 2 * genki;
        animator.Play(clips[currentClipIndex].name, 1, 0f);
    }

}
