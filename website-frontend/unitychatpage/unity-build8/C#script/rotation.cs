using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float mouseSensitivity = 100f;
    private float xRotation = 0f;
    private float yRotation = 0f;
    public float angle = 0f;
    private bool isCameraEnabled = false;  // カメラの動作状態

    void Update()
    {
        // カメラが有効な場合にのみ動作する
        if (isCameraEnabled)
        {
            float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity * Time.deltaTime;
            float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity * Time.deltaTime;

            xRotation -= mouseY;
            xRotation = Mathf.Clamp(xRotation, -90f, 90f);

            yRotation += mouseX;

            transform.localRotation = Quaternion.Euler(xRotation, yRotation + angle, 0f);
        }
    }

    // JavaScriptからのメッセージを受信する関数
    public void OnMessageReceived(string message)
    {
        if (message == "ENABLE_CAMERA")
        {
            isCameraEnabled = true;  // カメラ操作を有効化
        }
        else if (message == "DISABLE_CAMERA")
        {
            isCameraEnabled = false;  // カメラ操作を無効化
        }
    }
}
