using System.Collections;
using UnityEngine;
using TMPro;
using UnityEngine.Networking;
using System.Text;
using UnityEditor;


public class ChatFish : MonoBehaviour
{
    public TMP_InputField inputfield; // 入力フィールド
    public TextMeshPro talk1; // チャットの結果を表示
    public TextMeshPro talk2;
    public TextMeshPro talk3;
    public string character1;
    public string character2;
    public string character3;
    private string character;
    private string apiKey = "mykey";
    private string apiUrl = "myUrl";



    // APIへリクエストを送る処理
    public void StartComponent()
    {
        StartCoroutine(SendChatRequest(inputfield.text)); // 入力されたテキストを送信
    }

    // リクエスト送信
    private float requestCooldown = 15f; // 最低15秒の待機時間を設定
    private float lastRequestTime = 0f;

    // ChatGPT APIにリクエストを送信するコルーチン
    private IEnumerator SendChatRequest(string prompt)
    {
        // 最後のリクエストから一定時間が経過しているかチェック
        if (Time.time - lastRequestTime < requestCooldown)
        {
            Debug.Log("ちょっとまってね");
            talk1.text = "ちょっとまってね";
            talk2.text = "ちょっとまってね";
            talk3.text = "ちょっとまってね";
            yield break;
        }

        // リクエストを送信する前に時間を更新
        lastRequestTime = Time.time;
        if (GameObject.Find("FishV1-2(Clone)") != null) character = character1;
        else if(GameObject.Find("FishV2-2(Clone)")!=null) character = character2;
        else character = character3;

        string jsonString = "{ \"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"system\", \"content\": \""+ character +"\"}, {\"role\": \"user\", \"content\": \"" + prompt + "\"}], \"max_tokens\": 100 }";
        UnityWebRequest request = new UnityWebRequest(apiUrl, "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonString);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.SetRequestHeader("Authorization", "Bearer " + apiKey);

        // リクエストを送信
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            var response = JsonUtility.FromJson<ChatGPTResponse>(request.downloadHandler.text);
            talk1.text = response.choices[0].message.content;
            talk2.text = response.choices[0].message.content;
            talk3.text = response.choices[0].message.content;
        }
        else
        {
            talk1.text = "Error: " + request.error;
            talk2.text = "Error: " + request.error;
            talk3.text = "Error: " + request.error;
        }

        inputfield.text = ""; // 入力フィールドのクリア
    }
    public void ReceiveChatDataFromJS(string jsChat)
    {
        StartCoroutine(SendChatRequest(jsChat));
    }

    public void FinishFromJS()
    {
        talk1.text = "";
        talk2.text = "";
        talk3.text = "";
    }
    public void Chara1FromJS(string jsChara)
    {
        character1 = jsChara;
    }

    public void Chara2FromJS(string jsChara)
    {
        character2 = jsChara;
    }

    public void Chara3FromJS(string jsChara)
    {
        character3 = jsChara;
    }




    // ChatGPTの応答を保持するクラス
    [System.Serializable]
    public class ChatGPTResponse
    {
        public Choice[] choices;
    }

    [System.Serializable]
    public class Choice
    {
        public Message message;
    }

    [System.Serializable]
    public class Message
    {
        public string content;
    }
}
