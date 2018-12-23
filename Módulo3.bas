Attribute VB_Name = "Módulo3"
Sub Create_Pivot_Table_OC_Generada()
'Saber cuantas OC sourcing emitio cada comprador
'comparacion a nivel de llaves

Mensaje = MsgBox("Antes de crear esta tabla dinamica debe actualizar la planilla PO y Base" & vbCrLf & "A la planilla PO se agregara la columna Tipo de compra" & vbCrLf & "El filtro Tipo de compra indicara No Asignada cuando no este asignada" & vbCrLf & "Esta tabla dinamica indicara la fecha de creacion de la PO y la cantidad de PO creada", vbOKCancel)
If Mensaje = vbCancel Then
Exit Sub
End If


'Declare Variables
Dim PSheet As Worksheet 'To create a sheet for a new pivot table.
Dim DSheet As Worksheet 'To use as a data sheet.
Dim PCache As PivotCache 'To use as a name for pivot table cache.
Dim PTable As PivotTable 'To use as a name for our pivot table.
Dim PRange As Range 'to define source data range.
Dim lastRow As Long 'To get the last row and column of our data range.
Dim LastCol As Long 'To get the last row and column of our data range.

'Insert a New Blank Worksheet
On Error Resume Next
Application.DisplayAlerts = False
Worksheets("OC generada").Delete
Sheets.Add Before:=ActiveSheet
ActiveSheet.Name = "OC generada"
Application.DisplayAlerts = True
Set PSheet = Worksheets("OC generada")
Set DSheet = Worksheets("PO")

'Define Data Range
lastRow = DSheet.Cells(Rows.Count, 1).End(xlUp).Row
LastCol = DSheet.Cells(5, Columns.Count).End(xlToLeft).Column
Set PRange = DSheet.Cells(5, 1).Resize(lastRow, LastCol)

'Define Pivot Cache
Set PCache = ActiveWorkbook.PivotCaches.Create _
(SourceType:=xlDatabase, SourceData:=PRange). _
CreatePivotTable(TableDestination:=PSheet.Cells(2, 2), _
TableName:="OC_generada")

'Insert Blank Pivot Table
Set PTable = PCache.CreatePivotTable _
(TableDestination:=PSheet.Cells(1, 1), TableName:="OC_generada")

'Insert Report Filter (estos son los filtros)

With ActiveSheet.PivotTables("OC_generada").PivotFields("Tipo de compra")
.Orientation = xlPageField
'https://www.mrexcel.com/forum/excel-questions/593044-vba-hiding-pivotitems-when-pivotitem-count-10-a.html
.PivotItems("Sourcing").Visible = True
.PivotItems("Catalogo").Visible = True
.PivotItems("Politica").Visible = True
.PivotItems("(blank)").Visible = False
.Position = 2
End With


With ActiveSheet.PivotTables("OC_generada").PivotFields("Cantidad de lineas")
.Orientation = xlPageField
'https://www.mrexcel.com/forum/excel-questions/593044-vba-hiding-pivotitems-when-pivotitem-count-10-a.html
.PivotItems("1").Visible = True
        For i = 2 To .PivotItems.Count
            With .PivotItems(i)
                If i > 1 Then
                    .Visible = False
                End If
            End With
        Next i
.Position = 1
End With


'Insert Column Fields  (estos son las columnas)
With ActiveSheet.PivotTables("OC_generada").PivotFields("PO_DT")
.Orientation = xlColumnField
.PivotItems("(blank)").Visible = False
.Position = 1
End With

'Insert Row Labels (estos son las filas)
With ActiveSheet.PivotTables("OC_generada").PivotFields("Comprador")
.Orientation = xlRowField
.PivotItems("(blank)").Visible = False
.Position = 1
.Subtotals(1) = False
End With

'Insert Data Field (estos son las valores)
With ActiveSheet.PivotTables("OC_generada").PivotFields("Cantidad de lineas")
.Orientation = xlDataField
.Position = 1
.Function = xlCount
.NumberFormat = "#,##0"
'.Name = "Revenue " 'esto le cambia el nombre al campo de los valores
End With

'http://www.globaliconnect.com/excel/index.php?option=com_content&view=article&id=150:excel-pivot-tables-filter-data-items-values-a-dates-using-vba&catid=79&Itemid=475

'Format Pivot Table
'ActiveSheet.PivotTables("SalesPivotTable").ShowTableStyleRowStripes = True
'ActiveSheet.PivotTables("SalesPivotTable").TableStyle2 = "PivotStyleMedium9"

End Sub



