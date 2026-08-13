Require python3

Output docs/cli-demo.gif
Output docs/cli-demo.mp4

Set Shell bash
Set Width 1200
Set Height 720
Set FontSize 22
Set TypingSpeed 45ms
Set Theme "Catppuccin Frappe"
Set Padding 20
Set Margin 24
Set MarginFill "#303446"
Set WindowBar ColorfulRight
Set BorderRadius 8
Set CursorBlink false

Hide
Type `uv() { python3 demo/cli_demo.py; }; clear`
Enter
Show

Type "uv run python main.py"
Enter
Wait+Screen /Type your question/
Sleep 1s

Type "How is conversation history persisted between runs?"
Enter
Wait+Line /You:/
Sleep 3s

Type "exit"
Enter
Wait+Screen /Goodbye!/
Sleep 2s
