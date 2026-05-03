# Regras ProGuard para o app Selene Android
# WebView JS interface deve ser preservada
-keepclassmembers class com.selene.android.MainActivity {
    @android.webkit.JavascriptInterface <methods>;
}
-keep class com.selene.android.** { *; }
