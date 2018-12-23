Attribute VB_Name = "Módulo6"
Option Explicit
Sub Calcular_PO()
Dim Mensaje As Integer
Dim Ultimate_Row2 As Integer
Dim Ultimate_Column2 As Integer
Dim xCell As Variant
Dim i As Integer


Mensaje = MsgBox("Eliminara columnas, cambiara de formato las celdas de texto a numero y colocara la llave" & vbCrLf & "Los nombres de los campos debe estar en la fila 5, la informacion desde la fila 6", vbOKCancel)
If Mensaje = vbCancel Then
Exit Sub
End If

Worksheets("PO").Activate

Ultimate_Row2 = ActiveSheet.UsedRange.SpecialCells(xlCellTypeLastCell).Row
Ultimate_Column2 = ActiveSheet.UsedRange.SpecialCells(xlCellTypeLastCell).Column

'Eliminacion de columnas
Columns([45]).EntireColumn.Delete
Columns([44]).EntireColumn.Delete
Columns([43]).EntireColumn.Delete
Columns([42]).EntireColumn.Delete
Columns([41]).EntireColumn.Delete
Columns([40]).EntireColumn.Delete
Columns([39]).EntireColumn.Delete
Columns([38]).EntireColumn.Delete
Columns([37]).EntireColumn.Delete
Columns([36]).EntireColumn.Delete
Columns([35]).EntireColumn.Delete
Columns([34]).EntireColumn.Delete
Columns([33]).EntireColumn.Delete
Columns([32]).EntireColumn.Delete
Columns([31]).EntireColumn.Delete
Columns([30]).EntireColumn.Delete
Columns([29]).EntireColumn.Delete
Columns([28]).EntireColumn.Delete
Columns([27]).EntireColumn.Delete
Columns([26]).EntireColumn.Delete
Columns([25]).EntireColumn.Delete
Columns([24]).EntireColumn.Delete
Columns([23]).EntireColumn.Delete
Columns([22]).EntireColumn.Delete
Columns([21]).EntireColumn.Delete
Columns([20]).EntireColumn.Delete
Columns([19]).EntireColumn.Delete
Columns([18]).EntireColumn.Delete
Columns([17]).EntireColumn.Delete
Columns([16]).EntireColumn.Delete
Columns([14]).EntireColumn.Delete
Columns([13]).EntireColumn.Delete
Columns([12]).EntireColumn.Delete
Columns([10]).EntireColumn.Delete
Columns([9]).EntireColumn.Delete
Columns([8]).EntireColumn.Delete
'Columns("3:6").Delete
Range("C:C,D:D,E:E,F:F,Q:U").EntireColumn.Delete
'Columns([6]).EntireColumn.Delete
'Columns([5]).EntireColumn.Delete
'Columns([4]).EntireColumn.Delete
'Columns([3]).EntireColumn.Delete

'Cambiar el formato a los numeros para eliminar los ceros en la columna PO_Id
For Each xCell In Range(Cells(6, 2), Cells(Ultimate_Row2, 2))
    xCell.Value = CDec(xCell.Value)
Next xCell

'Cambiar el formato a los numeros para eliminar los ceros en la columna Req_Id
For Each xCell In Range(Cells(6, 3), Cells(Ultimate_Row2, 3))
    xCell.Value = CDec(xCell.Value)
Next xCell


'https://www.thoughtco.com/convert-text-to-number-in-excel-3424223
'https://www.microsofttraining.net/post-27242-vba-converting-text-numbers.html

'ordenar las columnas, debe quedar con Req_Id y PO_Id

Columns(3).Cut
Columns(2).Insert Shift:=xlToRight

'agregar una columna al inicio

Columns(1).Insert Shift:=xlToRight

'en esta columna concatenar BUSINESS_UNIT con REQ_ID

Cells(5, 1).Value = "Llave"
            Cells(5, 1).Font.Bold = True 'negrita
For i = 6 To Ultimate_Row2
    Cells(i, 1).Value = "= B" & i & "&" & "C" & i
Next i



'Cambiar el formato a los numeros para eliminar los ceros en la columna PO_Id

 'For i = 3 To Ultimate_Row
    'Cells(i, 4).Value = CDec(Cells(i, 4).Value)
 'Next i
 
'cruzar con Sol_Pen segun llave de BUSINESS_UNIT con REQ_ID
'el Business Unit de Sol_Pen cambiarlo a numero
' hacer la concatenacion y cruzar esta planilla con Sol_Pen

MsgBox "Contara las lineas correspondientes a cada solicitud"
Call countPO
MsgBox "Buscara el comprador que tiene asiganda la subcategoria, esto segun la planilla DATA "
Call encontrar_comprador
MsgBox "Buscara el tipo de compra segun planilla BASE "
Call tipo_de_compra_BASE

End Sub
Sub borrar_filas_PO()
Dim Mensaje As String
Mensaje = MsgBox("Limpiara todas las filas desde la fila 5 hasta la fila 8000", vbOKCancel)
If Mensaje = vbCancel Then
Exit Sub
End If
Worksheets("PO").Activate
Rows("5:8000").Delete
End Sub
'https://excelmacromastery.com/vba-dictionary/
'https://www.experts-exchange.com/articles/3391/Using-the-Dictionary-Class-in-VBA.html
'para contar las lineas de cada PO, esto es para ver cuantas PO hay en el listado
'https://stackoverflow.com/questions/915317/does-vba-have-dictionary-structure
'https://github.com/VBA-tools/VBA-Dictionary/blob/master/Dictionary.cls
Sub countPO()
Dim ws As Worksheet
Dim lastRow As Long, x As Long
Dim items As Object

Application.ScreenUpdating = False

Set ws = Worksheets("PO")
Cells(5, 8).Value = "Cantidad de lineas"
        Cells(5, 8).Font.Bold = True 'negrita
    
lastRow = ws.Range("A" & Rows.Count).End(xlUp).Row
    
    Set items = CreateObject("Scripting.Dictionary")
 
    For x = 6 To lastRow
    If Len(Cells(x, 2)) > 0 Then
        If Not items.Exists(ws.Range("A" & x).Value) Then
            items.Add ws.Range("A" & x).Value, 1
            ws.Range("H" & x).Value = items(ws.Range("A" & x).Value)
        Else
            items(ws.Range("A" & x).Value) = items(ws.Range("A" & x).Value) + 1
            ws.Range("H" & x).Value = items(ws.Range("A" & x).Value)
        End If
    End If
    Next x
    
  
End Sub
Sub encontrar_comprador()
Dim Ultimate_Row3 As Integer
Dim xCell As Variant
Dim i As Integer
'Categorizando hijas segun madres, con el comando Vlookup
'https://www.exceltrick.com/formulas_macros/vlookup-in-vba/
'para colocar el nombre del comprador
Cells(5, 7).Value = "Comprador"
        Cells(5, 7).Font.Bold = True 'negrita

Worksheets("PO").Activate
Ultimate_Row3 = ActiveSheet.UsedRange.SpecialCells(xlCellTypeLastCell).Row
'Ultimate_Column3 = ActiveSheet.UsedRange.SpecialCells(xlCellTypeLastCell).Column


'Cambiar el formato a los BUYER_ID para buscar
For Each xCell In Range(Cells(6, 6), Cells(Ultimate_Row3, 6))
    xCell.Value = CDec(xCell.Value)
Next xCell

For i = 6 To Ultimate_Row3
        Cells(i, 7) = Application.VLookup(Cells(i, 6), Worksheets("data").Range("D2:E50"), 2, False)
Next i

End Sub
Sub tipo_de_compra_BASE()
'https://www.exceltrick.com/formulas_macros/vlookup-in-vba/
Worksheets("PO").Activate
Dim Ultimate_Row3 As Integer
Dim i As Integer

Cells(5, 9).Value = "Tipo de compra"
            Cells(5, 9).Font.Bold = True 'negrita
For i = 6 To Ultimate_Row3
'si Cells(Ultimate_Column + 4) no hay caracteres y la celda "linea" > 1 entonces

    If Len(Cells(i, 2).Text) > 0 Then
        Cells(i, 9) = Application.VLookup(Cells(i, 3), Worksheets("Base").Range("AP6:AY10000"), 10, False)

            If Cells(i, 9).Text = "Sourcing" Then
                Cells(i, 9) = "Sourcing"
                ElseIf Cells(i, 9).Text = "Contrato" Then
                Cells(i, 9) = "Contrato"
                ElseIf Cells(i, 9).Text = "Catalogo" Then
                Cells(i, 9) = "Catalogo"
                Else 'If Cells(i, Ultimate_Column4 + 1).Text = "No Asignada" Then
                Cells(i, 9) = "No Asignada"
            End If
    End If
Next i
End Sub

Sub Reasignacion()
Dim Ultimate_Row6 As Integer
Dim Ultimate_Column6 As Integer
Dim j As Integer


Ultimate_Row6 = ActiveSheet.UsedRange.SpecialCells(xlCellTypeLastCell).Row
Ultimate_Column6 = ActiveSheet.UsedRange.SpecialCells(xlCellTypeLastCell).Column

Worksheets("Ayer").Activate
Cells(1, 1).Value = "Concatenar"
        Cells(1, 1).Font.Bold = True 'negrita
For j = 2 To Ultimate_Row6
    If Len(Cells(j, 4)) > 0 Then
    Cells(j, 1).Value = Cells(j, 2) & Cells(j, 3)
    End If
Next j


End Sub

Sub Limpiar_Reasignacion()
Worksheets("Ayer").Activate
Rows("2:8000").Clear
End Sub





 
 
 

