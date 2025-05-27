using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Linq;
using static FishSelect;

public class RoomSelect : MonoBehaviour
{
    public TMP_Dropdown dropdown;
    public Camera camera1;
    public Camera camera2;
    public Camera camera3;
    private List<string> roomNames = new List<string>() { "部屋1", "部屋2", "部屋3" };

    [System.Serializable]
    public class roomNamesWrapper
    {
        public string[] roomNames;  // 配列をラップするクラス
    }

    // Start is called before the first frame update
    void Start()
    {
        camera1.gameObject.SetActive(true);
        camera2.gameObject.SetActive(false);
        camera3.gameObject.SetActive(false);
    }

    // Update is called once per frame
    public void RoomChange()
    {
        string selectedText = dropdown.captionText.text;
        HandleSelection(selectedText);
    }
    private void HandleSelection(string selectedText)
    {
        if (selectedText == roomNames[0])
        {
            camera1.gameObject.SetActive(true);
            camera2.gameObject.SetActive(false);
            camera3.gameObject.SetActive(false);
        }
        else if (selectedText == roomNames[1])
        {
            camera1.gameObject.SetActive(false);
            camera2.gameObject.SetActive(true);
            camera3.gameObject.SetActive(false);
        }
        else if (selectedText == roomNames[2])
        {
            camera1.gameObject.SetActive(false);
            camera2.gameObject.SetActive(false);
            camera3.gameObject.SetActive(true);
        }
    }

    //jsとのデータの受け渡し
    public void ReceiveRoomDataFromJS(string roomNamesJson)
    {
        roomNamesWrapper roomNamesWrapper = JsonUtility.FromJson<roomNamesWrapper>(roomNamesJson);
        roomNames = roomNamesWrapper.roomNames.ToList();  // デシリアライズしてリストに変換
        for (int i = 0; i < 3; i++)
        {
            dropdown.options[i].text = roomNames[i];
        }
        dropdown.RefreshShownValue();
        
    }

}
